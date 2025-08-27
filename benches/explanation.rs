fn main() {
    eprintln!("Error: This project uses a custom benchmarking workflow.");
    eprintln!("Please choose a bench:");
    eprintln!("   Full Protocol Benches: 'cd ./benches/sumcheck-benches/ && cargo build --release && ./run_benches.sh'");
    eprintln!("   Lagrange Polynomial Benches: 'cd ./benches/lag-poly-benches/ && cargo build --release && ./run_benches.sh'");
    std::process::exit(1);
}
