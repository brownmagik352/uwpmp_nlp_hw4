import tensorflow as tf
from tensorflow.contrib.rnn import DropoutWrapper
from tensorflow.contrib.rnn import GRUCell
from tensorflow.nn import bidirectional_dynamic_rnn


def cbow_forward(config, inputs, scope=None):
    with tf.variable_scope(scope or "forward"):

        JX, JQ = config.max_context_size, config.max_ques_size
        d = config.hidden_size
        x, x_len, q, q_len = [inputs[key]
                              for key in ['x', 'x_len', 'q', 'q_len']]
        x_mask = tf.sequence_mask(x_len, JX)
        q_mask = tf.sequence_mask(q_len, JQ)

        # emb_mat = tf.get_variable('emb_mat', shape=[V, d])
        emb_mat = config.emb_mat_ph if config.serve else config.emb_mat
        emb_mat = tf.slice(emb_mat, [2, 0], [-1, -1])
        emb_mat = tf.concat(
            [tf.get_variable('emb_mat', shape=[2, d]), emb_mat], axis=0)
        xx = tf.nn.embedding_lookup(emb_mat, x, name='xx')  # [N, JX, d]
        qq = tf.nn.embedding_lookup(emb_mat, q, name='qq')  # [N, JQ, d]

        # equation 1 (averaging)
        qq_avg = tf.reduce_mean(
            bool_mask(qq, q_mask, expand=True), axis=1)  # [N, d]
        qq_avg_exp = tf.expand_dims(qq_avg, axis=1)  # [N, 1, d]
        qq_avg_tiled = tf.tile(qq_avg_exp, [1, JX, 1])  # [N, JX, d]

        xq = tf.concat([xx, qq_avg_tiled, xx * qq_avg_tiled],
                       axis=2)  # [N, JX, 3d]
        xq_flat = tf.reshape(xq, [-1, 3*d])  # [N * JX, 3*d]

        # Compute logits
        with tf.variable_scope('start'):
            logits1 = exp_mask(tf.reshape(tf.layers.dense(
                inputs=xq_flat, units=1), [-1, JX]), x_mask)  # [N, JX]
            yp1 = tf.argmax(logits1, axis=1)  # [N]
        with tf.variable_scope('stop'):
            logits2 = exp_mask(tf.reshape(tf.layers.dense(
                inputs=xq_flat, units=1), [-1, JX]), x_mask)  # [N, JX]
            yp2 = tf.argmax(logits2, axis=1)  # [N]

        outputs = {'logits1': logits1,
                   'logits2': logits2, 'yp1': yp1, 'yp2': yp2}
        variables = {'emb_mat': emb_mat}
        return variables, outputs


def rnn_forward(config, inputs, scope=None):
    with tf.variable_scope(scope or "rnn"):

        JX, JQ = config.max_context_size, config.max_ques_size
        d = config.hidden_size
        x, x_len, q, q_len = [inputs[key]
                              for key in ['x', 'x_len', 'q', 'q_len']]
        x_mask = tf.sequence_mask(x_len, JX)
        q_mask = tf.sequence_mask(q_len, JQ)

        # emb_mat = tf.get_variable('emb_mat', shape=[V, d])
        emb_mat = config.emb_mat_ph if config.serve else config.emb_mat
        emb_mat = tf.slice(emb_mat, [2, 0], [-1, -1])
        emb_mat = tf.concat(
            [tf.get_variable('emb_mat', shape=[2, d]), emb_mat], axis=0)
        xx = tf.nn.embedding_lookup(emb_mat, x, name='xx')  # [N, JX, d]
        qq = tf.nn.embedding_lookup(emb_mat, q, name='qq')  # [N, JQ, d]

        with tf.variable_scope("context_rnn"):

            # run context embeddings through GRU
            dropout = 0.1
            fw_cell = GRUCell(64)
            bw_cell = GRUCell(64)
            if config.is_train:
                fw_cell = DropoutWrapper(
                    fw_cell, output_keep_prob=(1.0 - dropout))
                bw_cell = DropoutWrapper(
                    bw_cell, output_keep_prob=(1.0 - dropout))
            (output_fw, output_bw), _ = bidirectional_dynamic_rnn(
                fw_cell, bw_cell, xx, dtype=tf.float32)
            xx_rnn_toobig = tf.concat([output_fw, output_bw], axis=2)
            xx_rnn = tf.layers.dense(xx_rnn_toobig, 50, activation=None)

        with tf.variable_scope("question_rnn"):

            # run question embeddings through GRU
            dropout = 0.1
            fw_cell2 = GRUCell(64)
            bw_cell2 = GRUCell(64)
            if config.is_train:
                fw_cell2 = DropoutWrapper(
                    fw_cell2, output_keep_prob=(1.0 - dropout))
                bw_cell2 = DropoutWrapper(
                    bw_cell2, output_keep_prob=(1.0 - dropout))
            (output_fw2, output_bw2), _ = bidirectional_dynamic_rnn(
                fw_cell2, bw_cell2, qq, dtype=tf.float32)
            qq_rnn_toobig = tf.concat([output_fw2, output_bw2], axis=2)
            qq_rnn = tf.layers.dense(qq_rnn_toobig, 50, activation=None)

        # equation 1 (averaging)
        qq_avg = tf.reduce_mean(
            bool_mask(qq_rnn, q_mask, expand=True), axis=1)  # [N, d]
        qq_avg_exp = tf.expand_dims(qq_avg, axis=1)  # [N, 1, d]
        qq_avg_tiled = tf.tile(qq_avg_exp, [1, JX, 1])  # [N, JX, d]

        xq = tf.concat([xx_rnn, qq_avg_tiled, xx_rnn * qq_avg_tiled],
                       axis=2)  # [N, JX, 3d]
        xq_flat = tf.reshape(xq, [-1, 3*d])  # [N * JX, 3*d]

        # Compute logits
        with tf.variable_scope('start'):
            logits1 = exp_mask(tf.reshape(tf.layers.dense(
                inputs=xq_flat, units=1), [-1, JX]), x_mask)  # [N, JX]
            yp1 = tf.argmax(logits1, axis=1)  # [N]
        with tf.variable_scope('stop'):
            logits2 = exp_mask(tf.reshape(tf.layers.dense(
                inputs=xq_flat, units=1), [-1, JX]), x_mask)  # [N, JX]
            yp2 = tf.argmax(logits2, axis=1)  # [N]

        outputs = {'logits1': logits1,
                   'logits2': logits2, 'yp1': yp1, 'yp2': yp2}
        variables = {'emb_mat': emb_mat}
        return variables, outputs


def attention_forward(config, inputs, scope=None):
    with tf.variable_scope(scope or "rnn_attention"):

        JX, JQ = config.max_context_size, config.max_ques_size
        d = config.hidden_size
        x, x_len, q, q_len = [inputs[key]
                              for key in ['x', 'x_len', 'q', 'q_len']]
        x_mask = tf.sequence_mask(x_len, JX)

        # emb_mat = tf.get_variable('emb_mat', shape=[V, d])
        emb_mat = config.emb_mat_ph if config.serve else config.emb_mat
        emb_mat = tf.slice(emb_mat, [2, 0], [-1, -1])
        emb_mat = tf.concat(
            [tf.get_variable('emb_mat', shape=[2, d]), emb_mat], axis=0)
        xx = tf.nn.embedding_lookup(emb_mat, x, name='xx')  # [N, JX, d]
        qq = tf.nn.embedding_lookup(emb_mat, q, name='qq')  # [N, JQ, d]

        with tf.variable_scope("context_rnn"):

            # run context embeddings through GRU
            dropout = 0.1
            fw_cell = GRUCell(64)
            bw_cell = GRUCell(64)
            if config.is_train:
                fw_cell = DropoutWrapper(
                    fw_cell, output_keep_prob=(1.0 - dropout))
                bw_cell = DropoutWrapper(
                    bw_cell, output_keep_prob=(1.0 - dropout))
            (output_fw, output_bw), _ = bidirectional_dynamic_rnn(
                fw_cell, bw_cell, xx, dtype=tf.float32, sequence_length=x_len)
            xx_rnn_toobig = tf.concat([output_fw, output_bw], axis=2)
            xx_rnn = tf.layers.dense(xx_rnn_toobig, 50, activation=None)

        with tf.variable_scope("question_rnn"):

            # run question embeddings through GRU
            dropout = 0.1
            fw_cell2 = GRUCell(64)
            bw_cell2 = GRUCell(64)
            if config.is_train:
                fw_cell2 = DropoutWrapper(
                    fw_cell2, output_keep_prob=(1.0 - dropout))
                bw_cell2 = DropoutWrapper(
                    bw_cell2, output_keep_prob=(1.0 - dropout))
            (output_fw2, output_bw2), _ = bidirectional_dynamic_rnn(
                fw_cell2, bw_cell2, qq, dtype=tf.float32, sequence_length=q_len)
            qq_rnn_toobig = tf.concat([output_fw2, output_bw2], axis=2)
            qq_rnn = tf.layers.dense(qq_rnn_toobig, 50, activation=None)

        # equation 10
        # how can i point-wise multiply xx_rnn and qq_rnn given their different sizes?
        xx_rnn_exp = tf.expand_dims(xx_rnn, axis=2)
        xx_rnn_tiled = tf.tile(xx_rnn_exp, [1, 1, JQ, 1])
        qq_rnn_exp = tf.expand_dims(qq_rnn, axis=1)
        qq_rnn_tiled = tf.tile(qq_rnn_exp, [1, JX, 1, 1])

        weights = tf.get_variable(name="weights", shape=[3*d, 1])
        bScalar = tf.get_variable(name="bScalar", shape=[])
        insideBrackets = tf.concat([xx_rnn_tiled, qq_rnn_tiled, tf.math.multiply(
            xx_rnn_tiled, qq_rnn_tiled)], axis=3)
        insideBracketsReshaped = tf.reshape(insideBrackets, [tf.shape(insideBrackets)[
            0] * tf.shape(insideBrackets)[1] * tf.shape(insideBrackets)[2], 3*d])
        dotProductWithWeightsPlusScalar = tf.matmul(
            insideBracketsReshaped, weights) + bScalar
        dotProductWithWeightsReshaped = tf.reshape(dotProductWithWeightsPlusScalar, [tf.shape(insideBrackets)[
            0], tf.shape(insideBrackets)[1], tf.shape(insideBrackets)[2]])
        p = tf.nn.softmax(dotProductWithWeightsReshaped, 2)

        p_exp = tf.expand_dims(p, axis=3)
        p_tiled = tf.tile(p_exp, [1, 1, 1, d])

        # equation 9
        qk_bar = tf.reduce_sum(tf.multiply(p_tiled, qq_rnn_tiled), axis=2)

        # plug qk_bar in place of qq_avg_tiled below

        xq = tf.concat([xx_rnn, qk_bar, xx_rnn * qk_bar],
                       axis=2)  # [N, JX, 3d]
        xq_flat = tf.reshape(xq, [-1, 3*d])  # [N * JX, 3*d]

        # Compute logits
        with tf.variable_scope('start'):
            logits1 = exp_mask(tf.reshape(tf.layers.dense(
                inputs=xq_flat, units=1), [-1, JX]), x_mask)  # [N, JX]
            yp1 = tf.argmax(logits1, axis=1)  # [N]
        with tf.variable_scope('stop'):
            logits2 = exp_mask(tf.reshape(tf.layers.dense(
                inputs=xq_flat, units=1), [-1, JX]), x_mask)  # [N, JX]
            yp2 = tf.argmax(logits2, axis=1)  # [N]

        outputs = {'logits1': logits1,
                   'logits2': logits2, 'yp1': yp1, 'yp2': yp2}
        variables = {'emb_mat': emb_mat}
        return variables, outputs


def get_loss(config, inputs, outputs, scope=None):
    with tf.name_scope(scope or "loss"):
        y1, y2 = inputs['y1'], inputs['y2']
        logits1, logits2 = outputs['logits1'], outputs['logits2']
        loss1 = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y1, logits=logits1))
        loss2 = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y2, logits=logits2))
        loss = loss1 + loss2
        acc1 = tf.reduce_mean(
            tf.cast(tf.equal(y1, tf.cast(tf.argmax(logits1, 1), 'int32')), 'float'))
        acc2 = tf.reduce_mean(
            tf.cast(tf.equal(y2, tf.cast(tf.argmax(logits2, 1), 'int32')), 'float'))
        tf.summary.scalar('loss', loss)
        tf.summary.scalar('acc1', acc1)
        tf.summary.scalar('acc2', acc2)
        return loss


def exp_mask(val, mask, expand=False):
    if expand:
        mask = tf.expand_dims(mask, -1)
    return val - (1.0 - tf.cast(mask, 'float')) * 10.0e10


def bool_mask(val, mask, expand=False):
    if expand:
        mask = tf.expand_dims(mask, -1)
    return val * tf.cast(mask, 'float')
