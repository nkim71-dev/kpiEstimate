import tensorflow as tf

# Dense Model
class Dense(tf.keras.Model):
    def __init__(self, inputDim, outputDim=1):
        super(Dense, self).__init__()
        # 모델
        self.dense1 = tf.keras.layers.Dense(64)
        self.lrelu1 = tf.keras.layers.ReLU(negative_slope=0.3)
        self.dropO2 = tf.keras.layers.Dropout(0.2)
        self.dense2 = tf.keras.layers.Dense(64)
        self.lrelu2 = tf.keras.layers.ReLU(negative_slope=0.3)
        self.dropO3 = tf.keras.layers.Dropout(0.2)
        self.dense3 = tf.keras.layers.Dense(64)
        self.lrelu3 = tf.keras.layers.ReLU(negative_slope=0.3)
        self.outputs = tf.keras.layers.Dense(outputDim)

    def call(self, inputs, training=None, mask=None):
        z = self.dense1(inputs)
        z = self.lrelu1(z)
        z = self.dropO2(z, training=training)
        z = self.dense2(z)
        z = self.lrelu2(z)
        z = self.dropO3(z, training=training)
        z = self.dense3(z)
        z = self.lrelu3(z)
        return self.outputs(z)

# Transformer Model
class Transformer(tf.keras.Model):
    def __init__(self, inputDim, outputDim=1):
        super(Transformer, self).__init__()
        # 모델
        self.reshape = tf.keras.layers.Reshape((1, inputDim))
        self.tran1 = TransformerBlock(1, 1, 128)
        self.tran2 = TransformerBlock(1, 1, 128)
        self.tran3 = TransformerBlock(1, 1, 128)
        self.reshape = tf.keras.layers.Reshape([-1,inputDim])
        self.flat = tf.keras.layers.Flatten()
        self.outputs = tf.keras.layers.Dense(outputDim)

    def call(self, inputs, training=None, mask=None):
        z = self.reshape(inputs)
        z = self.tran1(z, training=training)
        z = self.tran2(z, training=training)
        z = self.tran3(z, training=training)
        z = self.reshape(z)
        z = self.flat(z)
        return self.outputs(z)


# Transformer Network Blocks (Attention, Transformer)

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, embedding_dim, num_heads=1):
        super(MultiHeadAttention, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads

        assert embedding_dim % self.num_heads == 0

        self.projection_dim = embedding_dim // num_heads
        self.query_dense = tf.keras.layers.Dense(embedding_dim)
        self.key_dense = tf.keras.layers.Dense(embedding_dim)
        self.value_dense = tf.keras.layers.Dense(embedding_dim)
        self.dense = tf.keras.layers.Dense(embedding_dim)

    def create_padding_mask(x):
        mask = tf.cast(tf.math.equal(x, 0), tf.float32)
        # (batch_size, 1, 1, key의 문장 길이)
        return mask[:, tf.newaxis, tf.newaxis, :]

    def scaled_dot_product_attention(self, query, key, value, mask=None):
        matmul_qk = tf.matmul(query, key, transpose_b=True)
        depth =  tf.cast(tf.shape(key)[-1], tf.float32)
        logits = matmul_qk / tf.math.sqrt(depth)
        if mask is not None:
            logits += (mask * -1e9)
        attention_weights = tf.nn.softmax(logits, axis=-1)
        outputs = tf.matmul(attention_weights, value)
        outputs = tf.transpose(outputs, perm=[0,2,1,3])
        return outputs, attention_weights
    
    def split_head(self, x, batch_size):
        return tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
    
    def call(self, inputs, *args, **kwarg):
        batch_size = tf.shape(inputs)[0]

        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)

        query = self.split_head(query, batch_size)
        key = self.split_head(key, batch_size)
        value = self.split_head(value, batch_size)
        
        scaled_attention, attention_score = self.scaled_dot_product_attention(query, key, value)
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.embedding_dim))
        outputs = self.dense(concat_attention)

        return outputs

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embedding_dim, num_heads, dff, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadAttention(embedding_dim, num_heads)
        self.ffn = tf.keras.Sequential(
            [tf.keras.layers.Dense(dff, activation="relu"),
             tf.keras.layers.Dense(embedding_dim),]
        )
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs) # 첫번째 서브층 : 멀티 헤드 어텐션
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output) # Add & Norm
        ffn_output = self.ffn(out1) # 두번째 서브층 : 포지션 와이즈 피드 포워드 신경망
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output) # Add & Norm
    

# Predictor (안봐도 됨)
class Predictor(tf.keras.models.Model):
    def __init__(self, inputDim, outputDim=1, modelType='dense'):
        super(Predictor, self).__init__()
        # 모델 선언
        if 'transformer' in modelType.lower():
            self.model = Transformer(inputDim, outputDim)
        else:
            self.model = Dense(inputDim, outputDim)

        # 모델 Loss tracker
        self.lossTracker = tf.keras.metrics.Mean(name="loss")

    @property
    def metrics(self):
        return[self.lossTracker]
    
    def train_step(self, data):
        # 입력 데이터
        x,y = data
        
        # 학습
        with tf.GradientTape() as tape:
            # Feedforward & calc loss
            y_hat = self.model(x, training=True)
            totLoss = tf.math.reduce_mean((y-y_hat)**2)
 
        # Gradient 업데이트
        trainableVars = self.model.trainable_variables
        gradients = tape.gradient(totLoss, trainableVars)
        self.optimizer.apply_gradients(zip(gradients, trainableVars))

        # Loss tracking
        self.lossTracker.update_state(totLoss)

        return {"loss": self.lossTracker.result()}
    
    def test_step(self, data):
        # 입력 데이터
        x,y = data
        
        # Feedforward & calc loss
        y_hat = self.model(x, training=False)
        loss = tf.math.reduce_mean((y-y_hat)**2)
 
        # Loss tracking
        self.lossTracker.update_state(loss)

        return {"loss": self.lossTracker.result()}

    def call(self, inputs, training=False, mask=None):
        return self.model(inputs, training=training)

