# 2b2q

neural network to predict queue eta on 2b2t

## library or binary

this is both, the methods used to implement parsing and minor data analysis are
exposed, feel free to use _2b2q for analysing queue data csv files. for documentation
run `cargo doc` or just read the source code yourself

## commands

```man
USAGE:
    _2b2q <SUBCOMMAND>

OPTIONS:
    -h, --help    Print help information

SUBCOMMANDS:
    help     Print this message or the help of the given subcommand(s)
    new      creates a new neural network with specified layers at specified path, if using
                 `--dir` the model's file will be named after the layers
    stat     prints the current estimation of the specified models neatly organized to the
                 terminal
    train    trains the specified neural network on the data
```

### `_2b2q new`

```man
creates a new neural network with specified layers at specified path, if using
`--dir` the model's file will be named after the layers

USAGE:
    _2b2q new [OPTIONS] <--path <PATH>|--dir <DIR>> [LAYERS]...

ARGS:
    <LAYERS>...    layers which the neural net should have, for example: 10-6-2-4-1

OPTIONS:
    -d, --dir <DIR>      directory in which to place the model file
    -f, --force          force replacement of existing model file
    -h, --help           Print help information
    -p, --path <PATH>    path at which to place the model file
```

### `_2b2q stat`

```man
prints the current estimation of the specified models neatly organized to the terminal

USAGE:
    _2b2q stat <DATA_DIR> [MODELS]...

ARGS:
    <DATA_DIR>     directory from which to read `stat`ing data
    <MODELS>...    models which to include in comparison

OPTIONS:
    -h, --help    Print help information
```

### `_2b2q train`

```man
trains the specified neural network on the data

WARNING: changes apply immediately, make a backup if you are worried about it messing up

USAGE:
    _2b2q train [OPTIONS] <DATA_DIR> <MODEL>

ARGS:
    <DATA_DIR>
            directory from which to read training data

    <MODEL>
            

OPTIONS:
    -e, --epochs <EPOCHS>
            train specified amound of epochs per iteration

    -h, --help
            Print help information

    -l, --loop <loop>
            whether to loop after halt condition is reached conflicts with mse as error rate is not
            expected to decrease
            
            [default: true]

    -l, --logging <logging>
            enable or disable logging
            
            [default: true]

    -l, --logging-err-rate <LOGGING_ERR_RATE>
            whether to log error rate after how many batch iterations

    -m, --mse <MSE>
            train until specified error rate is achieved

    -m, --momentum <MOMENTUM>
            momentum used by RustNN (don't change without reason)
            
            [default: 0.1]

    -r, --rate <RATE>
            rate used for backpropagation by RustNN (don't change without reason)
            
            [default: 0.3]

    -t, --timer <TIMER>
            train for specified amount of seconds per iteration
```
