#[derive(Debug)]
pub enum FieldLabel {
    Field64,
    Field128,
    FieldBn254,
}

#[derive(Debug)]
pub enum AlgorithmLabel {
    CTY,
    VSBW,
    Blendy,
    ProductBlendy,
    ProductVSBW,
    ProductCTY,
}

pub struct BenchArgs {
    pub field_label: FieldLabel,
    pub algorithm_label: AlgorithmLabel,
    pub num_variables: usize,
    pub stage_size: usize,
}

fn check_if_number(input_string: String) -> bool {
    match input_string.parse::<usize>() {
        Ok(_) => true,
        Err(_) => false,
    }
}

pub fn validate_and_format_command_line_args(argsv: Vec<String>) -> BenchArgs {
    // Check if the correct number of command line arguments is provided
    if argsv.len() != 5 {
        eprintln!(
            "Usage: {} field_label algorithm_label num_variables stage_size",
            argsv[0]
        );
        std::process::exit(1);
    }
    // algorithm label
    if !(argsv[1] == "CTY"
        || argsv[1] == "VSBW"
        || argsv[1] == "Blendy"
        || argsv[1] == "ProductBlendy"
        || argsv[1] == "ProductVSBW"
        || argsv[1] == "ProductCTY")
    {
        eprintln!(
            "Usage: {} field_label algorithm_label num_variables stage_size",
            argsv[0]
        );
        eprintln!("Invalid input: algorithm_label must be one of (CTY, VSBW, Blendy, ProductVSBW, ProductBlendy, ProductCTY)");
        std::process::exit(1);
    }
    let algorithm_label = match argsv[1].as_str() {
        "CTY" => AlgorithmLabel::CTY,
        "VSBW" => AlgorithmLabel::VSBW,
        "Blendy" => AlgorithmLabel::Blendy,
        "ProductVSBW" => AlgorithmLabel::ProductVSBW,
        "ProductCTY" => AlgorithmLabel::ProductCTY,
        _ => AlgorithmLabel::ProductBlendy, // this is checked in previous line
    };
    // field_label
    if !(argsv[2] == "Field64" || argsv[2] == "Field128" || argsv[2] == "FieldBn254") {
        eprintln!(
            "Usage: {} field_label algorithm_label num_variables stage_size",
            argsv[0]
        );
        eprintln!("Invalid input: field_label must be one of (Field64, Field128, FieldBn254)");
        std::process::exit(1);
    }
    let field_label = match argsv[2].as_str() {
        "Field64" => FieldLabel::Field64,
        "Field128" => FieldLabel::Field128,
        _ => FieldLabel::FieldBn254, // this is checked in previous line
    };
    // num_variables
    if !check_if_number(argsv[3].clone()) {
        eprintln!(
            "Usage: {} field_label algorithm_label num_variables stage_size",
            argsv[0]
        );
        eprintln!("Invalid input: num_variables must be a number");
        std::process::exit(1);
    }
    let num_variables = argsv[3].clone().parse::<usize>().unwrap();
    // stage_size
    if !check_if_number(argsv[4].clone()) {
        eprintln!(
            "Usage: {} field_label algorithm_label num_variables stage_size",
            argsv[0]
        );
        eprintln!("Invalid input: stage_size must be a number");
        std::process::exit(1);
    }
    let stage_size = argsv[4].clone().parse::<usize>().unwrap();
    return BenchArgs {
        field_label,
        algorithm_label,
        num_variables,
        stage_size,
    };
}
