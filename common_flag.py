from tensorflow.python.platform import flags

FLAGS = flags.FLAGS
def define():
    flags.DEFINE_integer("batch_size", 32, "batch_size")
    flags.DEFINE_float("weight_decay",0.00004,"weight_decay")
    flags.DEFINE_float("learning_rate", 0.001, "learning_rate")
    flags.DEFINE_float("momentum", 0.9, "momentum")
    flags.DEFINE_float("clip_gradient_norm", 2.0, "clip_gradient_norm")
    flags.DEFINE_integer("num_char_classes",43,"number of char classes")
    flags.DEFINE_bool("is_training",True,"is training")
    flags.DEFINE_integer("seq_length", 26*2+1, "")
    flags.DEFINE_integer('num_lstm_units', 256,
                         'number of LSTM units for sequence LSTM')
    flags.DEFINE_float('lstm_state_clip_value', 10.0,
                       'cell state is clipped by this value prior to the cell'
                       ' output activation')
    flags.DEFINE_float('label_smoothing', 0,
                       'weight for label smoothing')
    flags.DEFINE_string("final_endpoint","Mixed_5d",'Endpoint to cut inception tower')
    flags.DEFINE_integer("img_width", 400, "")
    flags.DEFINE_integer("img_height", 64, "")
    flags.DEFINE_integer("img_channel", 3, "")
    flags.DEFINE_bool("ignore_nulls",True,"see nulls as normal label")
    flags.DEFINE_bool("average_across_timesteps",False,"LSTM config not know")
