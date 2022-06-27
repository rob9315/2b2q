use std::{
    fs::File,
    io::{BufWriter, Write},
    path::PathBuf,
};

use _2b2q::{
    nn::{log, make_expected_result, make_inputs},
    LoggingDataPoint,
};
use clap::{ArgGroup, Args, Parser, Subcommand};
use rayon::iter::{IntoParallelIterator, ParallelIterator};

#[derive(Parser)]
struct Cli {
    #[clap(subcommand)]
    sub: Commands,
}

#[derive(Subcommand)]
enum Commands {
    New(New),
    Stat(Stat),
    Train(Train),
}
#[derive(Args)]
#[clap(group = ArgGroup::new("file_path").required(true).multiple(false))]
/// creates a new neural network with specified layers at specified path,
/// if using `--dir` the model's file will be named after the layers
struct New {
    /// path at which to place the model file
    #[clap(short, long, conflicts_with = "dir", group = "file_path")]
    path: Option<PathBuf>,
    /// directory in which to place the model file
    #[clap(short, long, conflicts_with = "path", group = "file_path")]
    dir: Option<PathBuf>,
    /// force replacement of existing model file
    #[clap(short, long)]
    force: bool,
    /// layers which the neural net should have, for example: 10-6-2-4-1
    #[clap(value_delimiter('-'))]
    layers: Vec<u32>,
}
#[derive(Args)]
/// prints the current estimation of the specified models neatly organized
/// to the terminal
struct Stat {
    /// directory from which to read `stat`ing data
    data_dir: PathBuf,
    /// models which to include in comparison
    models: Vec<PathBuf>,
}
#[derive(Args)]
/// trains the specified neural network on the data
/// 
/// WARNING: changes apply immediately, make a backup if you are worried
/// about it messing up
#[clap(group = ArgGroup::new("halt_condition").required(false).multiple(false))]
struct Train {
    /// directory from which to read training data
    data_dir: PathBuf,
    model: PathBuf,
    /// whether to loop after halt condition is reached
    /// conflicts with mse as error rate is not expected to decrease
    #[clap(short, long, default_value_t = true, conflicts_with = "mse")]
    r#loop: bool,
    /// enable or disable logging
    #[clap(long, default_value_t = true)]
    logging: bool,
    /// whether to log error rate after how many batch iterations
    #[clap(long)]
    logging_err_rate: Option<u32>,
    /// train for specified amount of seconds per iteration
    #[clap(short, long, conflicts_with_all = &["epochs", "mse"], group = "halt_condition")]
    timer: Option<u64>,
    /// train specified amound of epochs per iteration
    #[clap(short, long, conflicts_with_all = &["timer", "mse"], group = "halt_condition")]
    epochs: Option<u32>,
    /// train until specified error rate is achieved
    #[clap(short, long, conflicts_with_all = &["epochs", "timer"], group = "halt_condition")]
    mse: Option<f64>,
    /// momentum used by RustNN (don't change without reason)
    #[clap(long, default_value_t = 0.1)]
    momentum: f64,
    /// rate used for backpropagation by RustNN (don't change without reason)
    #[clap(long, default_value_t = 0.3)]
    rate: f64,
}

fn main() {
    let opts = Cli::parse();

    match opts.sub {
        Commands::New(opts) => new(opts),
        Commands::Stat(opts) => stat(opts),
        Commands::Train(opts) => train(opts),
    }
}

fn new(opts: New) {
    if opts.dir.is_some() ^ opts.path.is_none() {
        eprintln!("exactly one of --path or --dir must be specified");
        std::process::exit(1);
    }

    let model = ::nn::NN::new(&opts.layers).to_json();

    fn layers_to_string(layers: &[u32]) -> String {
        use std::fmt::Write;
        let mut s = String::new();
        let mut layers = layers.iter();
        write!(&mut s, "{}", layers.next().unwrap()).ok();
        layers.for_each(|x| {
            write!(&mut s, "-{x}").ok();
        });
        s
    }

    let layers_string = layers_to_string(&opts.layers);

    let path = opts.path.unwrap_or_else(|| {
        let mut p = opts.dir.unwrap().join(&layers_string);
        p.set_extension("json");
        p
    });

    if !opts.force && path.exists() {
        eprintln!(
            "if you really want to overwrite the model at {:?}, pass --force as an option",
            path
        );
        std::process::exit(1);
    }

    if let Some(parent) = path.parent() {
        if !parent.exists() {
            std::fs::create_dir_all(parent).expect("failed to create parent directory");
        }
    }

    let mut file = std::fs::File::create(&path)
        .expect("something has gone wrong creating the file for your model");

    file.write_all(model.as_bytes())
        .expect("something has gone wrong writing the model to the file");

    let mut layers = opts.layers.iter();

    print!(
        "created model {:?} with layers {}",
        path,
        layers.next().unwrap()
    );
    layers.for_each(|l| print!("-{l}"));
    println!()
}
fn stat(opts: Stat) {
    let data =
        _2b2q::load_csv_dir(opts.data_dir).expect("problem loading data from supplied directory");

    let nets = opts
        .models
        .iter()
        .map(|path| (path.to_str().unwrap(), _2b2q::load_model(path)))
        .collect::<Vec<_>>();

    let logging_data_points = data
        .flatten()
        .map(|(x, p)| LoggingDataPoint::from_run(&x, p))
        .collect::<Vec<_>>();

    let borrowed = nets.iter().map(|x| (x.0, &x.1)).collect::<Vec<_>>();

    _2b2q::nn::log(&borrowed[..], &logging_data_points[..])
}
fn train(mut opts: Train) {
    if opts.mse.is_some() {
        opts.r#loop = false;
    }

    let mut net = _2b2q::load_model(&opts.model);

    let data =
        _2b2q::load_csv_dir(opts.data_dir).expect("problem loading data from supplied directory");

    let mut logging_data_points = vec![];
    let training_data_points: Vec<_> = {
        let mut training_runs = vec![];
        for (run, p) in data.flatten() {
            logging_data_points.push(LoggingDataPoint::from_run(&run, p));
            training_runs.extend(run);
        }
        training_runs
            .into_par_iter()
            .map(|point| (make_inputs(&point), make_expected_result(&point)))
            .collect()
    };

    let halt_condition = {
        use ::nn::HaltCondition::*;
        if let Some(mse) = opts.mse {
            MSE(mse)
        } else if let Some(epochs) = opts.epochs {
            Epochs(epochs)
        } else {
            Timer(
                chrono::Duration::from_std(std::time::Duration::from_secs(
                    opts.timer.unwrap_or(10),
                ))
                .expect("timer above limit"),
            )
        }
    };

    loop {
        if opts.logging {
            log(&[("new", &net)], &logging_data_points)
        }

        net.train(&training_data_points)
            .halt_condition(halt_condition)
            .log_interval(opts.logging_err_rate)
            .momentum(opts.momentum)
            .rate(opts.rate)
            .go();

        BufWriter::new(File::create(&opts.model).unwrap())
            .write_all(net.to_json().as_bytes())
            .ok();

        if !opts.r#loop {
            break;
        }
    }
}
