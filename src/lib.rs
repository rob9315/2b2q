use std::{
    fs::{File, ReadDir},
    io::{BufRead, BufReader, Lines},
    path::{Path, PathBuf},
    str::FromStr,
};

#[derive(Clone)]
pub struct QueueRun {
    pub start: QueueDataPoint,
    pub subsequent: Vec<QueueDataPoint>,
}
impl QueueRun {
    pub fn start_training_point(&self) -> TrainingDataPoint {
        self.start
            .with_start_and_end(self.start, *self.subsequent.last().unwrap())
    }
}

#[non_exhaustive]
#[derive(PartialEq, Clone, Copy)]
enum CsvHeaderItem {
    Time,
    Position,
    Length,
}
impl CsvHeaderItem {
    fn from_str(s: &str) -> Option<CsvHeaderItem> {
        use CsvHeaderItem::*;
        match &s.to_lowercase()[..] {
            "time" => Some(Time),
            "position" => Some(Position),
            "length" | "currentqueuelength" | "current_queue_length" => Some(Length),
            _ => None,
        }
    }
    fn vec_from_str<'a>(s: impl IntoIterator<Item = &'a str>) -> Option<Vec<CsvHeaderItem>> {
        let mut v = Vec::new();
        for mut s in s {
            if let Some(stripped) = s.strip_suffix('\n') {
                s = stripped;
            };
            let item = match CsvHeaderItem::from_str(s) {
                Some(item) => item,
                None => return None,
            };
            if v.contains(&item) {
                return None;
            }
            v.push(item);
        }
        Some(v)
    }
}
fn parse_csv_header(s: String) -> Option<Vec<CsvHeaderItem>> {
    CsvHeaderItem::vec_from_str(s.split(','))
}
impl QueueRun {
    fn from_csv_file(f: std::fs::File) -> Option<Self> {
        let mut reader = std::io::BufReader::new(f);
        let header = loop {
            let mut header_s = String::new();
            if reader.read_line(&mut header_s).is_err() || header_s.is_empty() {
                return None;
            };
            if let Some(header) = parse_csv_header(header_s) {
                break header;
            }
        };

        fn set_queue_data_point_item(
            i: CsvHeaderItem,
            QueueDataPoint {
                time,
                position,
                length,
            }: &mut QueueDataPoint,
            val: &str,
        ) -> Option<()> {
            match i {
                CsvHeaderItem::Time => FromStr::from_str(val).map(|x| *time = x).ok(),
                CsvHeaderItem::Position => FromStr::from_str(val).map(|x| *position = x).ok(),
                CsvHeaderItem::Length => FromStr::from_str(val).map(|x| *length = x).ok(),
            }
        }

        struct CsvQueueDataPointIterator {
            lines: Lines<BufReader<File>>,
            header: Vec<CsvHeaderItem>,
        }
        impl Iterator for CsvQueueDataPointIterator {
            type Item = QueueDataPoint;

            fn next(&mut self) -> Option<Self::Item> {
                let mut y = QueueDataPoint::default();
                while y == QueueDataPoint::default() {
                    match self.lines.next() {
                        Some(Ok(line)) => {
                            for (val, i) in line.split(',').zip(self.header.iter()) {
                                set_queue_data_point_item(*i, &mut y, val);
                            }
                        }
                        Some(Err(_)) => {}
                        None => return None,
                    }
                }

                Some(y)
            }
        }

        let mut iter = CsvQueueDataPointIterator {
            lines: reader.lines(),
            header,
        };

        let start = match iter.next() {
            Some(x) => x,
            None => return None,
        };
        Some(QueueRun {
            start,
            subsequent: iter.collect(),
        })
    }
}
impl IntoIterator for QueueRun {
    type Item = TrainingDataPoint;

    type IntoIter = QueueRunIterator;

    fn into_iter(self) -> Self::IntoIter {
        QueueRunIterator {
            inner: self,
            count: 0,
        }
    }
}
#[derive(Clone, Copy, Default, PartialEq)]
pub struct QueueDataPoint {
    pub time: u64,
    pub position: u16,
    pub length: u16,
}
impl QueueDataPoint {
    pub fn with_start_and_end(self, start: Self, end: Self) -> TrainingDataPoint {
        TrainingDataPoint {
            start_time: start.time,
            start_position: start.position,
            start_length: start.length,
            current_time: self.time,
            current_position: self.position,
            current_length: self.length,
            expected_output: end.time - self.time,
        }
    }
}
pub struct TrainingDataPoint {
    /// time at start in ms
    pub start_time: u64,
    /// queue position at start
    pub start_position: u16,
    /// queue length at start
    pub start_length: u16,
    /// time at snapshot in ms
    pub current_time: u64,
    /// queue position at snapshot
    pub current_position: u16,
    /// queue length at snapshot
    pub current_length: u16,
    /// time taken in ms
    pub expected_output: u64,
}

pub struct QueueRunIterator {
    inner: QueueRun,
    count: u64,
}
impl Iterator for QueueRunIterator {
    type Item = TrainingDataPoint;

    fn next(&mut self) -> Option<Self::Item> {
        let ret = match self.count {
            x if x > self.inner.subsequent.len() as u64 => return None,
            x if x == self.inner.subsequent.len() as u64 => self
                .inner
                .start
                .with_start_and_end(self.inner.start, *self.inner.subsequent.last().unwrap()),
            x => self.inner.subsequent[x as usize]
                .with_start_and_end(self.inner.start, *self.inner.subsequent.last().unwrap()),
        };
        self.count += 1;
        Some(ret)
    }
}

pub fn load_file(
    p: impl AsRef<Path>,
    x: fn(std::fs::File) -> Option<QueueRun>,
) -> Option<QueueRun> {
    std::fs::File::open(p.as_ref()).ok().and_then(x)
}

pub struct QueueDataDir {
    rd: ReadDir,
    x: fn(std::fs::File) -> Option<QueueRun>,
}
impl Iterator for QueueDataDir {
    type Item = Option<(QueueRun, PathBuf)>;

    fn next(&mut self) -> Option<Self::Item> {
        self.rd.next().map(|x| {
            x.ok()
                .and_then(|x| load_file(x.path(), self.x).map(|y| (y, x.path())))
        })
    }
}
pub fn load_dir(
    p: impl AsRef<Path>,
    x: fn(std::fs::File) -> Option<QueueRun>,
) -> std::io::Result<QueueDataDir> {
    std::fs::read_dir(p.as_ref()).map(|rd| QueueDataDir { rd, x })
}
pub fn load_csv_dir(p: impl AsRef<Path>) -> std::io::Result<QueueDataDir> {
    load_dir(p, QueueRun::from_csv_file)
}

const C: f64 = 150.0;

pub fn old_eta(current_pos: u16, queue_length: u16) -> f64 {
    let b = linear(queue_length.into()).ln();
    let a = |position: f64| -> f64 { ((position + C) / (queue_length as f64 + C)).ln() / b };
    a(0.0) - a(current_pos.into())
}

fn linear(point: f64) -> f64 {
    // dbg!(point);
    const VALUES: &[(f64, f64)] = &[
        (93.0, 0.9998618838664679),
        (207.0, 0.9999220416881794),
        (231.0, 0.9999234240704379),
        (257.0, 0.9999291667668093),
        (412.0, 0.9999410569845172),
        (418.0, 0.9999168965649361),
        (486.0, 0.9999440195022513),
        (506.0, 0.9999262577896301),
        (550.0, 0.9999462301738332),
        (586.0, 0.999938895110192),
        (666.0, 0.9999219189483673),
        (758.0, 0.9999473463335498),
        (789.0, 0.9999337457796981),
        (826.0, 0.9999279556964097),
    ];
    if point < VALUES[0].0 {
        // dbg!("lower than 93");
        return 0.0;
    }
    if point > VALUES.last().unwrap().0 {
        // dbg!("larger than 826");
        return VALUES.last().unwrap().1;
    }
    let (lower, higher) = VALUES
        .iter()
        .enumerate()
        .find(|(_, (x, _))| *x > point)
        .map(|(i, _)| (VALUES[i - 1], VALUES[i]))
        .unwrap();
    // dbg!(lower, higher);

    let a = (higher.1 - lower.1) / (higher.0 - lower.0);
    let b = -a * lower.0 + lower.1;
    a * point + b
}

pub fn load_model(path: impl AsRef<Path>) -> ::nn::NN {
    let mut s = String::default();
    std::io::Read::read_to_string(
        &mut File::open(&path).expect("model path doesn't exist"),
        &mut s,
    )
    .expect("failed reading model from file");
    ::nn::NN::from_json(&s)
}

pub struct LoggingDataPoint {
    file_path: PathBuf,
    pos: u16,
    len: u16,
    inputs: Vec<f64>,
    expected_time_h: f64,
    old_pred_h: f64,
}
impl LoggingDataPoint {
    pub fn from_run(run: &QueueRun, file_path: PathBuf) -> Self {
        let training_point = run.start_training_point();
        let inputs = nn::make_inputs(&training_point);
        let pos = run.start.position;
        let len = run.start.length;
        LoggingDataPoint {
            file_path,
            pos,
            len,
            expected_time_h: training_point.expected_output as f64 / 1000.0 / 3600.0,
            old_pred_h: old_eta(pos, len) / 3600.0,
            inputs,
        }
    }
}
pub mod nn {
    use chrono::{Datelike, NaiveDateTime, Timelike};

    use crate::{LoggingDataPoint, TrainingDataPoint};

    pub fn log(nets: &[(&str, &nn::NN)], data_points: &[LoggingDataPoint]) {
        let mut new: Vec<(&str, Vec<f64>)> = vec![];
        let mut old = vec![];
        for (_n, point) in data_points.iter().enumerate() {
            println!("#{_n} {}/{} {:?}", point.pos, point.len, &point.file_path);
            println!("pred\tdiff\tmodel");
            let old_pred_h = point.old_pred_h;
            let old_diff_minutes = (old_pred_h - point.expected_time_h) * 60.0;
            old.push(old_diff_minutes);
            for (n, (name, net)) in nets.iter().enumerate() {
                let result = net.run(&point.inputs)[0];
                let result_h = to_hours(result);
                let new_diff_minutes = (result_h - point.expected_time_h) * 60.0;
                new.get_mut(n)
                    .map(|x| x.1.push(new_diff_minutes))
                    .unwrap_or_else(|| new.push((name, vec![new_diff_minutes])));
                println!("{result_h:.2}h\t{}m\t{name}", new_diff_minutes.floor());
            }
            println!("{old_pred_h:.2}h\t{}m\told", old_diff_minutes.floor());
            println!("{:.2}h\t   \treal\n", point.expected_time_h);
        }
        fn abs(slice: &[f64]) -> f64 {
            slice.iter().map(|x| x.abs()).sum::<f64>() / slice.len() as f64
        }
        fn avg(slice: &[f64]) -> f64 {
            slice.iter().sum::<f64>() / slice.len() as f64
        }
        println!("abs\tavg\tmodel");
        for (name, new) in new {
            println!("{:.1}m\t{:.1}m\t{name}", abs(&new), avg(&new));
        }
        println!("{:.1}m\t{:.1}m\told\n", abs(&old), avg(&old));
    }

    fn inv_sigmoid(b: f64) -> f64 {
        -((1.0 / b) - 1.0).ln()
    }
    fn to_hours(b: f64) -> f64 {
        inv_sigmoid(b) * 14.0
    }
    fn nn_position(pos: u16) -> f64 {
        sigmoid(pos as f64 / 512.0)
    }
    fn _rev_position(pos: f64) -> u16 {
        (inv_sigmoid(pos) * 512.0) as u16
    }
    fn sigmoid(a: f64) -> f64 {
        1.0 / (1.0 + (-a).exp())
    }

    fn time(unix_millis: u64) -> NaiveDateTime {
        chrono::prelude::NaiveDateTime::from_timestamp(
            unix_millis as i64 / 1000,
            (unix_millis % 1000) as u32 * 1000,
        )
    }
    fn hour_of_day(time: NaiveDateTime) -> f64 {
        time.hour() as f64 / 23.0
    }
    fn day_of_week(time: NaiveDateTime) -> f64 {
        time.weekday() as u16 as f64 / 6.0
    }
    fn minute_of_hour(time: NaiveDateTime) -> f64 {
        time.minute() as f64 / 59.0
    }
    fn nn_sigmoid_queue_time(time: u64) -> f64 {
        sigmoid(time as f64 / 1000.0 / 3600.0 / 14.0)
    }
    pub fn make_inputs(point: &TrainingDataPoint) -> Vec<f64> {
        let TrainingDataPoint {
            start_time,
            start_position,
            start_length,
            current_time,
            current_position,
            current_length,
            ..
        } = point;

        let start = time(*start_time);
        let current = time(*current_time);

        vec![
            hour_of_day(start),
            day_of_week(start),
            minute_of_hour(start),
            nn_position(*start_position),
            nn_position(*start_length),
            hour_of_day(current),
            day_of_week(current),
            minute_of_hour(current),
            nn_position(*current_position),
            nn_position(*current_length),
        ]
    }
    pub fn make_expected_result(point: &TrainingDataPoint) -> Vec<f64> {
        vec![nn_sigmoid_queue_time(point.expected_output)]
    }
}

/*
const linear: <T extends number[], K extends number[]>(x: number | K, knownX: T, knownY: T) => K = require('everpolate').linear;

// function to get the eta from the current queue position and start position
// more information: https://github.com/themoonisacheese/2bored2wait/issues/141
export function eta(position: number, length: number) {
  const b = Math.log(linear(length, ...queueData)[0]);
  function a(position: number) {
    return Math.log((position + c) / (length + c)) / b;
  }
  return a(0) - a(position);
}
// assumed queue length, offset for queue eta calculation
const c = 150;
// statistically acquired data by MrGeorgen
// todo: get more
const queueData: [number[], number[]] = [
  [93, 207, 231, 257, 412, 418, 486, 506, 550, 586, 666, 758, 789, 826],
  [0.9998618838664679, 0.9999220416881794, 0.9999234240704379, 0.9999291667668093, 0.9999410569845172, 0.9999168965649361, 0.9999440195022513, 0.9999262577896301, 0.9999462301738332, 0.999938895110192, 0.9999219189483673, 0.9999473463335498, 0.9999337457796981, 0.9999279556964097],
]; */
