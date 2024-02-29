import tensorflow as tf
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, d_model, warmup_steps=4000):
    super().__init__()

    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)

    self.warmup_steps = warmup_steps

  def __call__(self, step):
    step = tf.cast(step, dtype=tf.float32)
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps ** -1.5)

    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
learning_rate = CustomSchedule(d_model=16)
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                     epsilon=1e-9)

class Pad_input(tf.keras.layers.Layer):
  def __init__(self,input_dim):
    super().__init__()
    self.start = tf.Variable(initial_value=np.array([0]*input_dim)[np.newaxis,:],trainable=False,shape=(1,input_dim),name='start_token')
    self.end = tf.Variable(initial_value=np.array([1]*input_dim)[np.newaxis,:],trainable=False,shape=(1,input_dim),name='end_token')
  def call(self,x):
    x_ext = np.array([np.concatenate([self.start,i.numpy()]) for i in x])
    x_ext = tf.ragged.constant(x_ext,ragged_rank=1)
    return x_ext
def positional_encoding(length, depth):
  depth = depth/2

  positions = np.arange(length)[:, np.newaxis]     # (seq, 1)
  depths = np.arange(depth)[np.newaxis, :]/depth   # (1, depth)

  angle_rates = 1 / (10000**depths)         # (1, depth)
  angle_rads = positions * angle_rates      # (pos, depth)

  pos_encoding = np.concatenate(
      [np.sin(angle_rads), np.cos(angle_rads)],
      axis=-1)

  return pos_encoding
class My_PositionalEmbedding(tf.keras.layers.Layer):
  def __init__(self,d_model):
    super().__init__()
    self.d_model = d_model
    self.pos_encoding = positional_encoding(length=5000, depth=d_model)
    self.dense = tf.keras.layers.Dense(d_model)

  def call(self, x):
    lengths = [i.shape[0] for i in x]
    x = self.dense(x)
    x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    positions = tf.ragged.constant([self.pos_encoding[:j,:] for j in lengths],ragged_rank=1)
    positions = tf.cast(positions,tf.float32)
    x = x + positions #positions could be better: create positon[MAXTOKENS,DEPTH] first. then position[i] = position[:len,depth]
    return x
class BaseAttention(tf.keras.layers.Layer):
  def __init__(self, **kwargs):
    super().__init__()
    self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
    self.layernorm = tf.keras.layers.LayerNormalization()
    self.add = tf.keras.layers.Add()
class CrossAttention(BaseAttention):
  def call(self, x, context):
    attn_output, attn_scores = self.mha(
        query=x,
        key=context,
        value=context,
        return_attention_scores=True)

    # Cache the attention scores for plotting later.
    self.last_attn_scores = attn_scores

    x = self.add([x, attn_output])
    x = self.layernorm(x)

    return x
class GlobalSelfAttention(BaseAttention):
  def call(self, x):
    attn_output = self.mha(
        query=x,
        value=x,
        key=x)
    x = self.add([x, attn_output])
    x = self.layernorm(x)
    return x
class CausalSelfAttention(BaseAttention):
  def call(self, x):
    attn_output = self.mha(
        query=x,
        value=x,
        key=x,
        use_causal_mask = True)
    x = self.add([x, attn_output])
    x = self.layernorm(x)
    return x
class FeedForward(tf.keras.layers.Layer):
  def __init__(self, d_model, dff, dropout_rate=0.1):
    super().__init__()
    self.seq = tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),
      tf.keras.layers.Dense(d_model),
      tf.keras.layers.Dropout(dropout_rate)
    ])
    self.add = tf.keras.layers.Add()
    self.layer_norm = tf.keras.layers.LayerNormalization()

  def call(self, x):
    x = self.add([x, self.seq(x)])
    x = self.layer_norm(x)
    return x

class EncoderLayer(tf.keras.layers.Layer):
  def __init__(self,*, d_model, num_heads, dff, dropout_rate=0.1):
    super().__init__()

    self.self_attention = GlobalSelfAttention(
        num_heads=num_heads,
        key_dim=d_model,
        dropout=dropout_rate)

    self.ffn = FeedForward(d_model, dff)

  def call(self, x):
    x = self.self_attention(x)
    x = self.ffn(x)
    return x
class My_Encoder(tf.keras.layers.Layer):
  def __init__(self, *, num_layers, d_model, num_heads,
               dff, vocab_size, dropout_rate=0.1):
    super().__init__()

    self.d_model = d_model
    self.num_layers = num_layers

    self.pos_embedding = My_PositionalEmbedding( d_model=d_model)

    self.enc_layers = [
        EncoderLayer(d_model=d_model,
                     num_heads=num_heads,
                     dff=dff,
                     dropout_rate=dropout_rate)
        for _ in range(num_layers)]
    self.dropout = tf.keras.layers.Dropout(dropout_rate)

  def call(self, x):
    # `x` is token-IDs shape: (batch, seq_len)
    x = self.pos_embedding(x)  # Shape `(batch_size, seq_len, d_model)`.

    # Add dropout.
    x = self.dropout(x)

    for i in range(self.num_layers):
      x = self.enc_layers[i](x)

    return x  # Shape `(batch_size, seq_len, d_model)`.
class DecoderLayer(tf.keras.layers.Layer):
  def __init__(self,
               *,
               d_model,
               num_heads,
               dff,
               dropout_rate=0.1):
    super(DecoderLayer, self).__init__()

    self.causal_self_attention = CausalSelfAttention(
        num_heads=num_heads,
        key_dim=d_model,
        dropout=dropout_rate)

    self.cross_attention = CrossAttention(
        num_heads=num_heads,
        key_dim=d_model,
        dropout=dropout_rate)

    self.ffn = FeedForward(d_model, dff)

  def call(self, x, context):
    x = self.causal_self_attention(x=x)
    x = self.cross_attention(x=x, context=context)

    # Cache the last attention scores for plotting later
    self.last_attn_scores = self.cross_attention.last_attn_scores

    x = self.ffn(x)  # Shape `(batch_size, seq_len, d_model)`.
    return x
class My_Decoder(tf.keras.layers.Layer):
  def __init__(self, *, num_layers, d_model, num_heads, dff, vocab_size,
               dropout_rate=0.1):
    super(My_Decoder, self).__init__()

    self.d_model = d_model
    self.num_layers = num_layers

    self.pos_embedding = My_PositionalEmbedding(d_model=d_model)
    self.dropout = tf.keras.layers.Dropout(dropout_rate)
    self.dec_layers = [
        DecoderLayer(d_model=d_model, num_heads=num_heads,
                     dff=dff, dropout_rate=dropout_rate)
        for _ in range(num_layers)]

    self.last_attn_scores = None

  def call(self, x, context):
    # `x` is token-IDs shape (batch, target_seq_len)
    x = self.pos_embedding(x)  # (batch_size, target_seq_len, d_model)

    x = self.dropout(x)

    for i in range(self.num_layers):
      x  = self.dec_layers[i](x, context)

    self.last_attn_scores = self.dec_layers[-1].last_attn_scores

    # The shape of x is (batch_size, target_seq_len, d_model).
    return x
class VAE_API(tf.keras.layers.Layer):
  def __init__(self,*,latent_dim):
    super().__init__()
    self.latent_dim = latent_dim
    self.dense = tf.keras.layers.Dense(latent_dim)
  def call(self,x):
    z_mean = self.dense(x)
    z_log_var = self.dense(x)
    return z_mean,z_log_var


class VAE_Encoder(tf.keras.layers.Layer):
  def __init__(self, *, num_layers, d_model, num_heads, dff,
               input_vocab_size, target_vocab_size, dropout_rate=0.1):
    super().__init__()
    self.encoder = My_Encoder(num_layers=num_layers, d_model=d_model,
                           num_heads=num_heads, dff=dff,
                           vocab_size=input_vocab_size,
                           dropout_rate=dropout_rate)
    self.api = VAE_API(latent_dim = d_model)
  def call(self,x):
    encoded = self.encoder(x)
    vae_encoded = self.api(encoded)
    return vae_encoded

class Sampler(tf.keras.layers.Layer):
    def call(self, z_mean, z_logvar):
        lens = [i.shape[0] for i in z_mean]
        max_len = np.max(lens)
        epsilon = tf.random.normal( shape=[z_mean.shape[0],max_len,z_mean.shape[2]] )
        epsilons = np.array([epsilon[i,:lens[i],:] for i in range(len(epsilon))])
        epsilons = np.array([i.numpy() for i in epsilons])
        epsilons = tf.ragged.constant(epsilons,ragged_rank=1)
        return z_mean + tf.exp(0.5 * z_logvar) * epsilons
    
class TVAE(tf.keras.Model):

    def __init__(self, *, num_layers, d_model, num_heads, dff,
               input_vocab_size, target_vocab_size, dropout_rate=0.1):
        super().__init__()
        self.encoder = VAE_Encoder(num_layers=num_layers, d_model=d_model,
                           num_heads=num_heads, dff=dff,
                           input_vocab_size = input_vocab_size,
                           target_vocab_size=target_vocab_size,
                           dropout_rate=dropout_rate)
        self.decoder = My_Decoder(num_layers=num_layers, d_model=d_model,
                           num_heads=num_heads, dff=dff,
                           vocab_size=target_vocab_size,
                           dropout_rate=dropout_rate)
        self.learning_rate = CustomSchedule(d_model=16)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                 epsilon=1e-9)
        self.d_model = d_model
        self.checkpoint_path = './training/cp.ckpt'
        self.sampler = Sampler()
        self.dense = tf.keras.layers.Dense(target_vocab_size, activation='softmax')
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")
        #self.pad = Pad_input(input_dim=input_vocab_size)

    @property
    def metrics(self): # list the metrics here to enable the model to reset them after each epoch
        return [self.total_loss_tracker, self.reconstruction_loss_tracker, self.kl_loss_tracker]
    
    def call(self,x):
        context = x
        dec_x = x[:,:-1,:]
        out = x[:,1:,:]
        z_mean, z_logvar = self.encoder(context)
        reconstruction = self.decoder(dec_x,z_mean)
        return reconstruction

    def train_step(self, x): # this is how you override the train_step() method of the keras.Model class
        #x = self.pad(x)
        context = x
        dec_x = x[:,:-1,:]
        out = x[:,1:,:]
        with tf.GradientTape() as tape:
            z_mean, z_logvar = self.encoder(context)
            z = self.sampler(z_mean, z_logvar)
            reconstruction = self.decoder(dec_x,z)
            reconstruction = self.dense(reconstruction)
           
            # Using 'auto'/'sum_over_batch_size' reduction type.
            mse = tf.keras.losses.MeanSquaredError()
            reconstruction_loss = mse(out, reconstruction)*self.d_model
            kl_loss = -0.5 * (1 + z_logvar - tf.square(z_mean) - tf.exp(z_logvar)) # K-L divergence for regularization
            total_loss = reconstruction_loss + tf.reduce_mean(kl_loss)
        grads = tape.gradient( total_loss, self.trainable_weights ) # retrieve the gradients; Use trainable_weights!
        self.optimizer.apply_gradients( zip(grads, self.trainable_variables) ) # apply_gradients() expects a list of (gradient, variable) pairs.
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "total_loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
    
    def get_latent_var(self, sentence):
        if len(sentence.shape)==2:
            sentence = sentence[np.newaxis,:,:]
        padding_shape = sentence.shape[-1]
        start_token = tf.zeros((1,1,padding_shape))
        encoded = self.encoder(sentence)[0]
        latent_vector = self.decoder(x=start_token,context=encoded)
        #latent_vector = tvae.dense(decoded)
        return tf.squeeze(latent_vector,axis=(0,1))
    
    def train_and_save(self, x, num_epochs=10):
        # Compile the mode
        checkpoint_path=self.checkpoint_path
        tvae.compile( optimizer=self.optimizer, run_eagerly=True )

        # Train the model
        for epoch in range(num_epochs):
            # Train for one epoch
            self.fit(x, epochs=1)
            
            # Save the model after every epoch
            self.save_weights(checkpoint_path.format(epoch=epoch))

    def load_checkpoint(self):
        checkpoint_path=self.checkpoint_path
        # Load model weights from the checkpoint file
        self.load_weights(checkpoint_path)
