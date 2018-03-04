#![feature(rustc_private)]

#[macro_use]
extern crate serde_derive;
extern crate serde;
extern crate csv;
extern crate rand;
extern crate tensorflow;

use rand::Rng;

use std::error::Error;
use std::fs::File;
use std::io::Read;
use std::result::Result;
use std::path::Path;
use std::process::exit;

use tensorflow::Code;
use tensorflow::Graph;
use tensorflow::ImportGraphDefOptions;
use tensorflow::Session;
use tensorflow::SessionOptions;
use tensorflow::Status;
use tensorflow::StepWithGraph;
use tensorflow::Tensor;

fn main() {
    exit(match run(201) {
        Ok(_) => 0,
        Err(e) => {
            println!("{}", e);
            1
        }
    })
}

/// Struct represeting the California Housing Train
/// dataset from the CSV file.
#[derive(Deserialize, Debug)]
struct CaliforniaHousing {
    longitude: Option<f32>,
    latitude: Option<f32>,
    housing_median_age: Option<f32>,
    total_rooms: Option<f32>,
    total_bedrooms: Option<f32>,
    population: Option<f32>,
    house_holds: Option<f32>,
    median_income: Option<f32>,
    median_house_value: Option<f32>,
}

fn setup_data() -> Result<Vec<CaliforniaHousing>, Box<Error>> {
    let filename = "data/california-housing-train.csv";
    if !Path::new(filename).exists() {
        return Err(Box::new(
            Status::new_set(Code::NotFound, &format!("File {} not found!", filename)).unwrap(),
        ));
    }
    println!("Found file {}!", filename);

    // unwrap because we know that the file exists here.
    let mut reader = csv::Reader::from_path(filename).unwrap();
    let mut houses: Vec<CaliforniaHousing> = reader.deserialize::<CaliforniaHousing>().filter_map(
        |e| -> Option<CaliforniaHousing> {
            // scale the median house value to be in units of thousands,
            // so it can be learned a little more easily with learning rates in a range that we usually use.
            if let Ok(mut entry) = e {
                if let Some(median_house_value) = entry.median_house_value {
                    entry.median_house_value = Some(median_house_value / 1000f32);
                }
                Some(entry)
            } else {
                None
            }
        },
    ).collect();

    // Randomize the data, just to be sure not to get any pathological ordering
    // effects that might harm the performance of Stochastic Gradient Descent
    let mut rng = rand::thread_rng();
    rng.shuffle(houses.as_mut_slice());

    Ok(houses)
}

fn run(num_steps: usize) -> Result<(), Box<Error>> {
    let data = setup_data()?;
    let mut feature_total_rooms: Tensor<f32> = Tensor::new(&[data.len() as u64]);
    let mut target_median_room_value: Tensor<f32> = Tensor::new(&[data.len() as u64]);
    for (i, value) in data.into_iter().enumerate() {
        feature_total_rooms[i] = if let Some(total_rooms) = value.total_rooms {
            total_rooms
        } else {
            0f32
        };
        target_median_room_value[i] = if let Some(median_house_value) = value.median_house_value {
            median_house_value
        } else {
            0f32
        };
    }

    let mut graph = Graph::new();
    let mut proto = Vec::new();
    File::open("data/models/model_ex1.pb")?.read_to_end(&mut proto)?;
    graph.import_graph_def(&proto, &ImportGraphDefOptions::new())?;
    let mut session = Session::new(&SessionOptions::new(), &graph)?;
    let op_x = graph.operation_by_name_required("x")?;
    let op_y = graph.operation_by_name_required("y")?;
    let op_init = graph.operation_by_name_required("init")?;
    let op_train = graph.operation_by_name_required("train")?;
    let op_w = graph.operation_by_name_required("w")?;
    let op_b = graph.operation_by_name_required("b")?;

    let mut init_step = StepWithGraph::new();
    init_step.add_input(&op_x, 0, &feature_total_rooms);
    init_step.add_input(&op_y, 0, &target_median_room_value);
    init_step.add_target(&op_init);
    session.run(&mut init_step)?;

    let mut train_step = StepWithGraph::new();
    train_step.add_input(&op_x, 0, &feature_total_rooms);
    train_step.add_input(&op_y, 0, &target_median_room_value);
    train_step.add_target(&op_train);
    for _ in 0..num_steps {
        session.run(&mut train_step)?;
    }

    let mut output_step = StepWithGraph::new();
    let w_ix = output_step.request_output(&op_w, 0);
    let b_ix = output_step.request_output(&op_b, 0);
    session.run(&mut output_step)?;

    let w_hat: f32 = output_step.take_output(w_ix)?[0];
    let b_hat: f32 = output_step.take_output(b_ix)?[0];

    let w = 0.1;
    println!("Checking w: expected {}, got {}, {}",
        w,
        w_hat,
        if (w - w_hat).abs() < 1e-3 {
            "Success!"
        } else {
            "FAIL"
        });
    let b = 0.3;
    println!("Checking b: expected {}, got {}, {}",
        b,
        b_hat,
        if (b - b_hat).abs() < 1e-3 {
            "Success!"
        } else {
            "FAIL"
        });

    Ok(())
}
