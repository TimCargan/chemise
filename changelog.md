# 0.0.3
- Added a new keep_best callback and early stopping framework
  - `on_epoch_end` callback functions can now raise an `EarlyStopping` Exception to halt training
- Improvements to the way LSTM and autoreg LSTM works
- Add a nan to num gard on the grads in basic trainer