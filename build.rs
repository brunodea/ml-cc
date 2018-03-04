use std::process::Command;

fn main() {
    Command::new("python3").arg("scripts/gen_proto_ex1.py")
        .status().unwrap();
}