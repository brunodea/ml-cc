extern crate serde;
#[macro_use]
extern crate serde_derive;

extern crate csv;
extern crate tensorflow;

use std::error::Error;
use std::io;
use std::process;
use std::result::Result;
use std::path::Path;
use std::process::exit;

use tensorflow::Code;
use tensorflow::Status;

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

fn run() -> Result<(), Box<Error>> {
    let filename = "data/california-housing-train.csv";
    if !Path::new(filename).exists() {
        return Err(Box::new(
            Status::new_set(Code::NotFound, &format!("File {} not found!", filename)).unwrap(),
        ));
    }
    println!("Found file {}!", filename);

    // unwrap because we know that the file exists here.
    let mut rdr = csv::Reader::from_path(filename).unwrap();
    for result in rdr.deserialize() {
        let entry: CaliforniaHousing = result?;
        println!("{:?}", entry);
    }

    Ok(())
}
