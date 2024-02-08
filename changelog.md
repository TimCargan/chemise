# 0.0.2

- Added a new keep_best callback and early stopping framework
  - `on_epoch_end` callback functions can now raise an `EarlyStopping` Exception to halt training
- Vector trainer now supports calls to the `__call__` method so can be used for inference with `model(data)`
- Fixed bug for vector trainer logging
- `Checkpointer` updated to use new orbax checkpointing
- Add a nan to num gard on the grads in basic trainer
- New Data interface and list data type
  - Added a new `data` interface to support no `tensorflow.data` datasets. The interface is compatible and existing data pipelines should still work fine 
  - Remove tensorflow has been as a core dependency
- Improvements to the way LSTM and autoreg LSTM works
- Improved test coverage 
- Better docstrings and flak8 

# 0.0.1-alpha
- Initial version published alongside [local-global-solar](https://github.com/TimCargan/local-global-solar)
