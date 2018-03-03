#![feature(rustc_private)]

#[macro_use]
extern crate serde_derive;
extern crate serde;
extern crate csv;
extern crate rand;
extern crate tensorflow;

use rand::Rng;

use std::error::Error;
use std::result::Result;
use std::path::Path;
use std::process::exit;

use tensorflow::Code;
use tensorflow::Status;
use tensorflow::Tensor;

fn main() {
    exit(match run() {
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
    longitude: Option<f64>,
    latitude: Option<f64>,
    housing_median_age: Option<f64>,
    total_rooms: Option<f64>,
    total_bedrooms: Option<f64>,
    population: Option<f64>,
    house_holds: Option<f64>,
    median_income: Option<f64>,
    median_house_value: Option<f64>,
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
                    entry.median_house_value = Some(median_house_value / 1000f64);
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

fn run() -> Result<(), Box<Error>> {
    let data = setup_data()?;
    let mut feature_total_rooms: Tensor<f64> = Tensor::new(&[data.len() as u64]);
    let mut target_median_room_value: Tensor<f64> = Tensor::new(&[data.len() as u64]);
    for (i, value) in data.into_iter().enumerate() {
        feature_total_rooms[i] = if let Some(total_rooms) = value.total_rooms {
            total_rooms
        } else {
            0f64
        };
        target_median_room_value[i] = if let Some(median_house_value) = value.median_house_value {
            median_house_value
        } else {
            0f64
        };
    }

    Ok(())
}
