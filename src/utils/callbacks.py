import os
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard, Callback, ModelCheckpoint, ReduceLROnPlateau # type: ignore

# def get_callbacks(train_config):
#     return [
#         TensorBoard(log_dir=train_config['log_path']),
#         ModelCheckpoint(filepath=os.path.join(train_config['model_save_path'], 'best_model.keras'), monitor='val_loss', save_best_only=True),
#         ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
#     ]

from tensorflow.keras.callbacks import Callback, TensorBoard, ModelCheckpoint # type: ignore

class CustomReduceLROnPlateau(Callback):
    def __init__(self, monitor='val_loss', factor=0.5, patience=5, min_lr=0, **kwargs):
        super(CustomReduceLROnPlateau, self).__init__(**kwargs)
        self.monitor = monitor
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.best = float('inf')
        self.wait = 0
        self.lr_epsilon = tf.keras.backend.epsilon()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = logs.get(self.monitor)

        if current is None:
            return

        if current < self.best:
            self.best = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                old_lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
                if old_lr > self.min_lr + self.lr_epsilon:
                    new_lr = max(old_lr * self.factor, self.min_lr)
                    self.model.optimizer.learning_rate.assign(new_lr)
                    print(f"\nEpoch {epoch + 1}: ReduceLROnPlateau reducing learning rate to {new_lr}.")
                self.wait = 0

def get_callbacks(train_config):
    return [
        TensorBoard(log_dir=train_config['log_path']),
        ModelCheckpoint(filepath=os.path.join(train_config['model_save_path'], 'best_model.keras'), monitor='val_loss', save_best_only=True),
        CustomReduceLROnPlateau(factor=0.5, patience=5)
    ]

# import tensorflow as tf
# from tensorflow.keras.callbacks import Callback, TensorBoard, ModelCheckpoint # type: ignore

# class CustomReduceLROnPlateau(Callback):
#     def __init__(self, initial_lr, factor=0.5, patience=5, min_lr=0, **kwargs):
#         super(CustomReduceLROnPlateau, self).__init__(**kwargs)
#         self.initial_lr = initial_lr
#         self.factor = factor
#         self.patience = patience
#         self.min_lr = min_lr
#         self.best = float('inf')
#         self.wait = 0
#         self.lr_epsilon = tf.keras.backend.epsilon()

#     def on_epoch_end(self, epoch, logs=None):
#         logs = logs or {}
#         current = logs.get('val_loss')

#         if current is None:
#             return

#         if current < self.best:
#             self.best = current
#             self.wait = 0
#         else:
#             self.wait += 1
#             if self.wait >= self.patience:
#                 old_lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
#                 new_lr = max(old_lr * self.factor, self.min_lr)
                
#                 # Create a new learning rate schedule with the updated learning rate
#                 new_lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
#                     initial_learning_rate=new_lr,
#                     decay_steps=100000,
#                     decay_rate=0.96,
#                     staircase=True
#                 )

#                 # Update the model's optimizer with the new learning rate schedule
#                 self.model.optimizer.learning_rate = new_lr_schedule
#                 print(f"\nEpoch {epoch + 1}: ReduceLROnPlateau reducing learning rate to {new_lr}.")
#                 self.wait = 0

# def get_callbacks(train_config):
#     initial_lr = train_config['learning_rates']['generator']  # Use the initial learning rate
#     return [
#         TensorBoard(log_dir=train_config['log_path']),
#         ModelCheckpoint(filepath=os.path.join(train_config['model_save_path'], 'best_model.keras'), monitor='val_loss', save_best_only=True),
#         CustomReduceLROnPlateau(initial_lr=initial_lr, factor=0.5, patience=5)
#     ]


# class CustomReduceLROnPlateau(Callback):
#     def __init__(self, monitor='val_loss', factor=0.5, patience=5, min_lr=0, **kwargs):
#         super(CustomReduceLROnPlateau, self).__init__(**kwargs)
#         self.monitor = monitor
#         self.factor = factor
#         self.patience = patience
#         self.min_lr = min_lr
#         self.best = float('inf')
#         self.wait = 0
#         self.lr_epsilon = tf.keras.backend.epsilon()
    
#     def on_epoch_end(self, epoch, logs=None):
#         logs = logs or {}
#         current = logs.get(self.monitor)
        
#         if current is None:
#             return
        
#         if current < self.best:
#             self.best = current
#             self.wait = 0
#         else:
#             self.wait += 1
#             if self.wait >= self.patience:
#                 old_lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
#                 if old_lr > self.min_lr + self.lr_epsilon:
#                     new_lr = max(old_lr * self.factor, self.min_lr)
#                     # tf.keras.backend.set_value(self.model.optimizer.learning_rate, new_lr)
#                     self.model.optimizer.learning_rate = new_lr
#                     print(f"\nEpoch {epoch + 1}: ReduceLROnPlateau reducing learning rate to {new_lr}.")
#                 self.wait = 0

# def get_callbacks(train_config):
#     return [
#         TensorBoard(log_dir=train_config['log_path']),
#         ModelCheckpoint(filepath=os.path.join(train_config['model_save_path'], 'best_model.keras'), monitor='val_loss', save_best_only=True),
#         CustomReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
#     ]
