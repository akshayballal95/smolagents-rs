use std::{any::Any, collections::HashMap, fmt};

use rustpython_parser::{
    ast::{
        self,
        bigint::{BigInt, Sign},
        Constant, Expr, Operator, Stmt, UnaryOp,
    },
    Parse,
};

// Custom error type for interpreter
#[derive(Debug)]
pub enum InterpreterError {
    SyntaxError(String),
    RuntimeError(String),
    OperationLimitExceeded,
    UnauthorizedImport(String),
    UnsupportedOperation(String),
}

impl fmt::Display for InterpreterError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            InterpreterError::SyntaxError(msg) => write!(f, "Syntax Error: {}", msg),
            InterpreterError::RuntimeError(msg) => write!(f, "Runtime Error: {}", msg),
            InterpreterError::OperationLimitExceeded => write!(
                f,
                "Operation limit exceeded. Possible infinite loop detected."
            ),
            InterpreterError::UnauthorizedImport(module) => {
                write!(f, "Unauthorized import of module: {}", module)
            }
            InterpreterError::UnsupportedOperation(op) => {
                write!(f, "Unsupported operation: {}", op)
            }
        }
    }
}

type ToolFunction = Box<dyn Fn(Vec<Constant>) -> Result<Constant, InterpreterError>>;

fn setup_static_tools() -> HashMap<String, ToolFunction> {
    let mut tools = HashMap::new();

    // Basic type conversions
    tools.insert(
        "float".to_string(),
        Box::new(|args: Vec<Constant>| {
            if args.len() != 1 {
                return Err(InterpreterError::RuntimeError(
                    "float() takes exactly one argument".to_string(),
                ));
            }
            Ok(Constant::Float((&args[0]).clone().float().unwrap_or(0.0)))
        }) as ToolFunction,
    );

    tools.insert(
        "int".to_string(),
        Box::new(|args: Vec<Constant>| {
            if args.len() != 1 {
                return Err(InterpreterError::RuntimeError(
                    "int() takes exactly one argument".to_string(),
                ));
            }
            Ok(Constant::Int(BigInt::from(
                (&args[0]).clone().float().unwrap_or(0.0) as i64,
            )))
        }) as ToolFunction,
    );

    tools.insert(
        "bool".to_string(),
        Box::new(|args: Vec<Constant>| {
            if args.len() != 1 {
                return Err(InterpreterError::RuntimeError(
                    "bool() takes exactly one argument".to_string(),
                ));
            }
            Ok(Constant::Bool(
                !matches!(&args[0], Constant::Bool(false))
                    && !matches!(&args[0],
                        Constant::Int(i) if *i == BigInt::from(0)
                    ),
            ))
        }) as ToolFunction,
    );

    // Math functions
    tools.insert(
        "abs".to_string(),
        Box::new(|args: Vec<Constant>| {
            if args.len() != 1 {
                return Err(InterpreterError::RuntimeError(
                    "abs() takes exactly one argument".to_string(),
                ));
            }
            Ok(Constant::Float(
                (&args[0]).clone().float().unwrap_or(0.0).abs(),
            ))
        }) as ToolFunction,
    );

    tools.insert(
        "round".to_string(),
        Box::new(|args: Vec<Constant>| {
            if args.len() != 1 && args.len() != 2 {
                return Err(InterpreterError::RuntimeError(
                    "round() takes one or two arguments".to_string(),
                ));
            }
            let num = (&args[0]).clone().float().unwrap_or(0.0);
            let digits = if args.len() == 2 {
                (&args[1]).clone().float().unwrap_or(0.0) as i32
            } else {
                0
            };
            let factor = 10.0f64.powi(digits);
            Ok(Constant::Float((num * factor).round() / factor))
        }) as ToolFunction,
    );

    // Aggregation functions
    tools.insert(
        "sum".to_string(),
        Box::new(|args: Vec<Constant>| {
            Ok(Constant::Float(
                args.iter().map(|a| a.clone().float().unwrap_or(0.0)).sum(),
            ))
        }) as ToolFunction,
    );

    tools.insert(
        "max".to_string(),
        Box::new(|args: Vec<Constant>| {
            if args.is_empty() {
                return Err(InterpreterError::RuntimeError(
                    "max() arg is an empty sequence".to_string(),
                ));
            }
            Ok(Constant::Float(
                args.iter()
                    .map(|a| a.clone().float().unwrap_or(0.0))
                    .fold(f64::NEG_INFINITY, f64::max),
            ))
        }) as ToolFunction,
    );

    tools.insert(
        "min".to_string(),
        Box::new(|args: Vec<Constant>| {
            if args.is_empty() {
                return Err(InterpreterError::RuntimeError(
                    "min() arg is an empty sequence".to_string(),
                ));
            }
            Ok(Constant::Float(
                args.iter()
                    .map(|a| a.clone().float().unwrap_or(0.0))
                    .fold(f64::INFINITY, f64::min),
            ))
        }) as ToolFunction,
    );

    // Length function
    tools.insert(
        "len".to_string(),
        Box::new(|args: Vec<Constant>| {
            if args.len() != 1 {
                return Err(InterpreterError::RuntimeError(
                    "len() takes exactly one argument".to_string(),
                ));
            }
            match &args[0] {
                Constant::Tuple(t) => Ok(Constant::Int(BigInt::from(t.len()))),
                _ => Err(InterpreterError::RuntimeError(
                    "object has no len()".to_string(),
                )),
            }
        }) as ToolFunction,
    );

    tools
}

fn main() {
    let static_tools = setup_static_tools();

    // let python_source = r#"
    // def is_odd(i):
    //     return bool(i & 1)
    // p = is_odd(1)
    // "#;
    let python_source = r#"
a,b = 2 + 3, 4
c= 3.14
sum(a+1,b)
abs(-1)
round(c)
    "#;
    let mut state = HashMap::new();
    let ast = ast::Suite::parse(python_source, "<embedded>").unwrap();
    println!("{:?}", ast);
    evaluate_ast(&ast, &mut state, &static_tools).unwrap();

    // assert!(tokens.all(|t| t.is_ok()));
}

fn evaluate_ast(
    ast: &ast::Suite,
    state: &mut HashMap<String, Box<dyn Any>>,
    static_tools: &HashMap<
        String,
        Box<dyn Fn(Vec<Constant>) -> Result<Constant, InterpreterError>>,
    >,
) -> Result<(), InterpreterError> {
    for node in ast.iter() {
        match node {
            Stmt::FunctionDef(func) => {
                println!("{:?}", func.name);
            }
            Stmt::Expr(expr) => {
                evaluate_expr(&expr.value, state, static_tools)?;
            }

            Stmt::Assign(assign) => {
                for target in assign.targets.iter() {
                    // let target = evaluate_expr(&Box::new(target.clone()), state, static_tools)?;
                    match target {
                        ast::Expr::Name(name) => {
                            let value = evaluate_expr(&assign.value, state, static_tools)?;
                            state.insert(name.id.to_string(), Box::new(value));
                        }
                        ast::Expr::Tuple(target_names) => {
                            let value = evaluate_expr(&assign.value, state, static_tools)?;
                            let values = value.tuple().ok_or_else(|| {
                                InterpreterError::RuntimeError(format!(
                                    "Tuple unpacking failed. Expected values of type tuple",
                                ))
                            })?;
                            if target_names.elts.len() != values.len() {
                                return Err(InterpreterError::RuntimeError(format!(
                                    "Tuple unpacking failed. Expected {} values, got {}",
                                    target_names.elts.len(),
                                    values.len()
                                )));
                            }
                            for (i, target_name) in target_names.elts.iter().enumerate() {
                                match target_name {
                                    ast::Expr::Name(name) => {
                                        state.insert(
                                            name.id.to_string(),
                                            Box::new(values[i].clone()),
                                        );
                                    }
                                    _ => panic!("Expected string"),
                                }
                            }
                        }
                        _ => panic!("Expected string"),
                    }
                }
                println!("State: a{:?}", state["a"].downcast_ref::<Constant>());
                println!("State: b{:?}", state["b"].downcast_ref::<Constant>());
            }

            _ => {}
        }
    }
    Ok(())
}

fn convert_bigint_to_f64(i: &BigInt) -> f64 {
    let i = i.to_u32_digits();
    let num = i.1.iter().fold(0i64, |acc, &d| acc * (1 << 32) + d as i64);
    match i.0 {
        Sign::Minus => -num as f64,
        Sign::NoSign | Sign::Plus => num as f64,
    }
}

// fn evaluate_numeric_constant(
//     constant: &Box<ast::Constant>,
//     state: &mut HashMap<String, Box<dyn Any>>,
// ) -> Result<f64, InterpreterError> {
//     match &**constant {
//         Constant::Int(i) => Ok(convert_bigint_to_f64(&i)),
//         Constant::Float(f) => Ok(*f),
//         Constant::Str(s) => {
//             let value = state
//                 .get(s)
//                 .ok_or_else(|| {
//                     InterpreterError::RuntimeError(format!(
//                         "Variable '{}' used before assignment",
//                         s
//                     ))
//                 })?
//                 .downcast_ref::<Constant>()
//                 .ok_or_else(|| {
//                     InterpreterError::RuntimeError(format!("Expected numeric value. Got {:?}", s))
//                 })?;
//             evaluate_numeric_constant(&Box::new(value.clone()), state)
//         }
//         _ => panic!("Expected numeric value"),
//     }
// }

fn evaluate_expr(
    expr: &Box<Expr>,
    state: &mut HashMap<String, Box<dyn Any>>,
    static_tools: &HashMap<
        String,
        Box<dyn Fn(Vec<Constant>) -> Result<Constant, InterpreterError>>,
    >,
) -> Result<Constant, InterpreterError> {
    match &**expr {
        ast::Expr::Call(call) => {
            let func = match &*call.func {
                ast::Expr::Name(name) => name.id.to_string(),
                ast::Expr::Attribute(attr) => {
                    let func = evaluate_expr(&Box::new(*attr.value.clone()), state, static_tools)?;
                    let func = func.str().unwrap();
                    func
                }
                _ => panic!("Expected function name"),
            };
            let args = call
                .args
                .iter()
                .map(|e| evaluate_expr(&Box::new(e.clone()), state, static_tools))
                .collect::<Result<Vec<Constant>, InterpreterError>>()?;
            println!("Function: {:?}", func);
            println!("Args: {:?}", args);
            if static_tools.contains_key(&func) {
                println!("Static tool");
                let result = static_tools[&func](args);
                println!("Result: {:?}", result);
                result
            } else {
                Ok(Constant::Int(BigInt::from(1)))
            }
        }
        ast::Expr::BinOp(binop) => {
            let left_val = evaluate_expr(&binop.left.clone(), state, static_tools)?
                .float()
                .unwrap();
            let right_val = evaluate_expr(&binop.right.clone(), state, static_tools)?
                .float()
                .unwrap();
            match &binop.op {
                Operator::Add => {
                    println!("{} + {} = {}", left_val, right_val, left_val + right_val);
                    Ok(Constant::Float(left_val + right_val))
                }
                Operator::Sub => {
                    println!("{} - {} = {}", left_val, right_val, left_val - right_val);
                    Ok(Constant::Float(left_val - right_val))
                }
                Operator::Mult => {
                    println!("{} * {} = {}", left_val, right_val, left_val * right_val);
                    Ok(Constant::Float(left_val * right_val))
                }
                Operator::Div => {
                    println!("{} / {} = {}", left_val, right_val, left_val / right_val);
                    Ok(Constant::Float(left_val / right_val))
                }
                Operator::FloorDiv => {
                    println!("{} // {} = {}", left_val, right_val, left_val / right_val);
                    Ok(Constant::Float(left_val / right_val))
                }
                Operator::Mod => {
                    println!("{} % {} = {}", left_val, right_val, left_val % right_val);
                    Ok(Constant::Float(left_val % right_val))
                }
                Operator::Pow => {
                    println!(
                        "{} ** {} = {}",
                        left_val,
                        right_val,
                        left_val.powf(right_val)
                    );
                    Ok(Constant::Float(left_val.powf(right_val)))
                }
                Operator::BitOr => {
                    println!("{}", 1 | 0);

                    println!(
                        "{} | {} = {}",
                        left_val,
                        right_val,
                        left_val as i64 | right_val as i64
                    );
                    Ok(Constant::Int(BigInt::from(
                        left_val as i64 | right_val as i64,
                    )))
                }
                Operator::BitXor => {
                    println!(
                        "{} ^ {} = {}",
                        left_val,
                        right_val,
                        left_val as i64 ^ right_val as i64
                    );
                    Ok(Constant::Int(BigInt::from(
                        left_val as i64 ^ right_val as i64,
                    )))
                }
                Operator::BitAnd => {
                    println!(
                        "{} & {} = {}",
                        left_val,
                        right_val,
                        left_val as i64 & right_val as i64
                    );
                    Ok(Constant::Int(BigInt::from(
                        left_val as i64 & right_val as i64,
                    )))
                }
                Operator::LShift => {
                    let left_val = left_val as i64;
                    let right_val = right_val as i64;
                    println!("{} << {} = {}", left_val, right_val, left_val << right_val);
                    Ok(Constant::Int(BigInt::from(left_val << right_val)))
                }
                Operator::RShift => {
                    let left_val = left_val as i64;
                    let right_val = right_val as i64;
                    println!("{} >> {} = {}", left_val, right_val, left_val >> right_val);
                    Ok(Constant::Int(BigInt::from(left_val >> right_val)))
                }
                Operator::MatMult => {
                    println!("{} * {} = {}", left_val, right_val, left_val * right_val);
                    Ok(Constant::Float(left_val * right_val))
                }
            }
        }
        ast::Expr::UnaryOp(unaryop) => {
            let operand = evaluate_expr(&unaryop.operand, state, static_tools)?;
            match &unaryop.op {
                UnaryOp::USub => {
                    if let Constant::Float(f) = operand {
                        Ok(Constant::Float(-f))
                    } else {
                        panic!("Expected float")
                    }
                }
                UnaryOp::UAdd => Ok(operand),
                UnaryOp::Not => {
                    if let Constant::Bool(b) = operand {
                        Ok(Constant::Bool(!b))
                    } else {
                        panic!("Expected boolean")
                    }
                }
                UnaryOp::Invert => {
                    if let Constant::Float(f) = operand {
                        Ok(Constant::Float(-(f as i64) as f64))
                    } else {
                        panic!("Expected float")
                    }
                }
            }
        }
        ast::Expr::Constant(constant) => match &constant.value {
            Constant::Int(i) => Ok(Constant::Float(convert_bigint_to_f64(&i))),
            _ => Ok(constant.value.clone()),
        },
        ast::Expr::List(list) => Ok(Constant::Tuple(
            list.elts
                .iter()
                .map(|e| evaluate_expr(&Box::new(e.clone()), state, static_tools))
                .collect::<Result<Vec<Constant>, InterpreterError>>()?,
        )),
        ast::Expr::Name(name) => {
            if state.contains_key(&name.id.to_string()) {
                Ok(state[&name.id.to_string()]
                    .downcast_ref::<Constant>()
                    .unwrap()
                    .clone())
            } else {
                Err(InterpreterError::RuntimeError(format!(
                    "Variable '{}' used before assignment",
                    name.id
                )))
            }
        }
        ast::Expr::Tuple(tuple) => Ok(Constant::Tuple(
            tuple
                .elts
                .iter()
                .map(|e| evaluate_expr(&Box::new(e.clone()), state, static_tools))
                .collect::<Result<Vec<Constant>, InterpreterError>>()?,
        )),
        _ => {
            panic!("Unsupported expression: {:?}", expr);
        }
    }
}
