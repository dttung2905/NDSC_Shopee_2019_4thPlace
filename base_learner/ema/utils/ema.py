class ExponentialMovingAverage(Callback):
    def __init__(self, model, decay=0.999, mode='epoch', n=100):
        """
        mode: 'epoch': Do update_weights every epoch.
              'batch':                   every n batches.
        n   :
        """
        self.decay = decay
        self.mode = mode
        self.ema_model = clone_model(model)
        self.ema_model.set_weights(model.get_weights())
        self.n = n
        if self.mode is 'batch':
            self.cnt = 0
        self.ema_weights = [K.get_value(w) for w in model.trainable_weights]
        self.n_weights = len(self.ema_weights)
        super(ExponentialMovingAverage, self).__init__()

    def on_batch_end(self, batch, logs={}):
        if self.mode is 'batch':
            self.cnt += 1
            if self.cnt % self.n == 0:
                self.update_weights()

    def on_epoch_end(self, epoch, logs={}):
        if self.mode is 'epoch':
            self.update_weights()
        for var, w in zip(self.ema_model.trainable_weights, self.ema_weights):
            K.set_value(var, w)

    def update_weights(self):
        for w_old, var_new in zip(self.ema_weights, self.model.trainable_weights):
            w_old += (1 - self.decay) * (K.get_value(var_new) - w_old)


if __name__ == '__main__':
    print('running EMA model')
