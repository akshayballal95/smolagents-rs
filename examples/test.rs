use pyo3::types::{PyDict, PyModule};
use pyo3::{prelude::*, prepare_freethreaded_python};
use rustpython_parser::{
    ast::{
        self,
        bigint::{BigInt, Sign},
        Constant, Expr, Operator, Stmt, UnaryOp,
    },
    Parse,
};
use serde_json;
use smolagents_rs::tools::{DuckDuckGoSearchTool, Tool};
use std::{any::Any, collections::HashMap, fmt};

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

impl From<PyErr> for InterpreterError {
    fn from(err: PyErr) -> Self {
        InterpreterError::RuntimeError(err.to_string())
    }
}

#[derive(Clone, Debug)]
pub enum CustomConstant {
    Int(BigInt),
    Float(f64),
    Str(String),
    Bool(bool),
    Tuple(Vec<Constant>),
    PyObj(PyObject),  // Add this variant
}

impl CustomConstant {
    pub fn float(&self) -> Option<f64> {
        match self {
            CustomConstant::Float(f) => Some(*f),
            _ => None,
        }
    }
    pub fn str(&self) -> Option<String> {
        match self {
            CustomConstant::Str(s) => Some(s.clone()),
            _ => None,
        }
    }
    pub fn tuple(&self) -> Option<Vec<Constant>> {
        match self {
            CustomConstant::Tuple(t) => Some(t.clone()),
            _ => None,
        }
    }
}

impl From<CustomConstant> for Constant {
    fn from(custom: CustomConstant) -> Self {
        match custom {
            CustomConstant::Int(i) => Constant::Int(i),
            CustomConstant::Float(f) => Constant::Float(f),
            CustomConstant::Str(s) => Constant::Str(s),
            CustomConstant::Bool(b) => Constant::Bool(b),
            CustomConstant::Tuple(t) => Constant::Tuple(t),
            CustomConstant::PyObj(_) => {
                panic!("PyObj is not supported in Constant");
            }
        }
    }
}

impl From<Constant> for CustomConstant {
    fn from(constant: Constant) -> Self {
        match constant {
            Constant::Int(i) => CustomConstant::Int(i),
            Constant::Float(f) => CustomConstant::Float(f),
            Constant::Str(s) => CustomConstant::Str(s),
            Constant::Bool(b) => CustomConstant::Bool(b),
            Constant::Tuple(t) => CustomConstant::Tuple(t),
            _ => panic!("Unsupported constant type"),
        }
    }
}

type ToolFunction = Box<dyn Fn(Vec<Constant>) -> Result<CustomConstant, InterpreterError>>;


fn setup_static_tools() -> HashMap<String, ToolFunction> {
    let mut tools = HashMap::new();

    let function_map: HashMap<&str, &str> = [
        ("float", "float"),
        ("int", "int"),
        ("bool", "bool"),
        ("str", "str"),
        ("abs", "abs"),
        ("round", "round"),
        ("sum", "sum"),
        ("max", "max"),
        ("min", "min"),
        ("len", "len"),
        ("ord", "ord"),
        ("chr", "chr"),
        ("enumerate", "enumerate"),
        ("type", "type"),
        ("iter", "iter"),
        ("range", "range"),
        ("reversed", "reversed"),
        // Math module functions
        ("ceil", "math.ceil"),
        ("floor", "math.floor"),
        ("sqrt", "math.sqrt"),
        ("sin", "math.sin"),
        ("cos", "math.cos"),
        ("tan", "math.tan"),
        ("exp", "math.exp"),
        ("log", "math.log"),
        ("acos", "math.acos"),
        ("asin", "math.asin"),
        ("atan", "math.atan"),
        ("atan2", "math.atan2"),
        ("degrees", "math.degrees"),
        ("radians", "math.radians"),
        ("pow", "math.pow"),
    ]
    .iter()
    .cloned()
    .collect();
    let function_map_clone = function_map.clone();

    let eval_py = move |func: &str, args: Vec<Constant>| {
        Python::with_gil(|py| {
            let locals = PyDict::new(py);

            // Import required modules
            let math = PyModule::import(py, "math")?;
            locals.set_item("math", math)?;

            for (i, arg) in args.iter().enumerate() {
                match arg {
                    Constant::Float(f) => locals.set_item(format!("arg{}", i), f)?,
                    Constant::Int(i) => {
                        locals.set_item(format!("arg{}", i), convert_bigint_to_f64(i))?
                    }
                    Constant::Str(s) => locals.set_item(format!("arg{}", i), s)?,
                    Constant::Tuple(t) => {
                        let py_list: Vec<f64> =
                            t.iter().map(|x| x.clone().float().unwrap_or(0.0)).collect();
                        locals.set_item(format!("arg{}", i), py_list)?
                    }
                    _ => locals.set_item(format!("arg{}", i), 0.0)?,
                }
            }

            let arg_names: Vec<String> = (0..args.len()).map(|i| format!("arg{}", i)).collect();
            let func_path = function_map_clone.get(func).unwrap_or(&"builtins.float");
            let expr = format!("{}({})", func_path, arg_names.join(","));
            println!("Evaluating: {}", expr);

            let result = py.eval(&expr, None, Some(locals))?;
            println!("Result: {:?}", result);
            // Handle different return types
            if let Ok(float_val) = result.extract::<f64>() {
                Ok(CustomConstant::Float(float_val))
            } else if let Ok(string_val) = result.extract::<String>() {
                Ok(CustomConstant::Str(string_val))
            } else if let Ok(bool_val) = result.extract::<bool>() {
                Ok(CustomConstant::Bool(bool_val))
            } 
            else {
                Ok(CustomConstant::PyObj(result.into_py(py)))
                // Err(InterpreterError::RuntimeError("Unsupported return type".to_string()))
            }
        })
    };

    // Register tools after eval_py is defined
    for func in function_map.keys() {
        let func = func.to_string(); // Create owned String
        let eval_py = eval_py.clone(); // Clone the closure
        tools.insert(
            func.clone(),
            Box::new(move |args| eval_py(&func, args)) as ToolFunction,
        );
    }

    tools
}

fn setup_custom_tools(tools: Vec<Box<dyn Tool>>) -> HashMap<String, ToolFunction> {
    let mut tools_map = HashMap::new();
    for tool in tools {
        tools_map.insert(
            tool.name().to_string(),
            Box::new(move |args: Vec<Constant>| {
                if let Some(Constant::Str(query)) = args.first() {
                    let tool = DuckDuckGoSearchTool::new();
                    match tool.forward(&query) {
                        Ok(results) => {
                            let json = serde_json::to_string_pretty(&results).unwrap_or_default();
                            Ok(CustomConstant::Str(json))
                        }
                        Err(e) => Ok(CustomConstant::Str(format!("Error: {}", e))),
                    }
                } else {
                    Ok(CustomConstant::Str("Error: Invalid arguments".to_string()))
                }
            }) as ToolFunction,
        );
    }
    tools_map
}

fn main() {
    prepare_freethreaded_python();
    let static_tools = setup_static_tools();
    let custom_tools = setup_custom_tools(vec![Box::new(DuckDuckGoSearchTool::new())]);
    // let python_source = r#"
    // def is_odd(i):
    //     return bool(i & 1)
    // p = is_odd(1)
    // "#;
    let python_source = r#"
a,b = 2.2 + 3, 4
c= 3.14
sum([a+1,b])
len([1,2,3])
enumerate([1,2,3])
pow(2,3)
type(2)
sqrt(3.4)

    "#;
    let mut state = HashMap::new();
    let ast = ast::Suite::parse(python_source, "<embedded>").unwrap();
    println!("{:?}", ast);
    evaluate_ast(&ast, &mut state, &static_tools, &custom_tools).unwrap();

    // assert!(tokens.all(|t| t.is_ok()));
}

fn evaluate_ast(
    ast: &ast::Suite,
    state: &mut HashMap<String, Box<dyn Any>>,
    static_tools: &HashMap<
        String,
        Box<dyn Fn(Vec<Constant>) -> Result<CustomConstant, InterpreterError>>,
    >,
    custom_tools: &HashMap<String, ToolFunction>,
) -> Result<(), InterpreterError> {
    for node in ast.iter() {
        match node {
            Stmt::FunctionDef(func) => {
                println!("{:?}", func.name);
            }
            Stmt::Expr(expr) => {
                evaluate_expr(&expr.value, state, static_tools, custom_tools)?;
            }
            Stmt::For(for_stmt) => {
                println!("For: {:?}", for_stmt.iter);
                let iter = evaluate_expr(&for_stmt.iter.clone(), state, static_tools, custom_tools)?;
                println!("Iter: {:?}", iter);
                let iter = iter.tuple().unwrap();
                for item in iter {
                    state.insert("i".to_string(), Box::new(CustomConstant::from(item)));
                    // evaluate_expr(&for_stmt.body.clone(), state, static_tools, custom_tools)?;
                }
            }

            Stmt::Assign(assign) => {
                for target in assign.targets.iter() {
                    // let target = evaluate_expr(&Box::new(target.clone()), state, static_tools)?;
                    match target {
                        ast::Expr::Name(name) => {
                            let value =
                                evaluate_expr(&assign.value, state, static_tools, custom_tools)?;
                            state.insert(name.id.to_string(), Box::new(CustomConstant::from(value)));
                        }
                        ast::Expr::Tuple(target_names) => {
                            let value =
                                evaluate_expr(&assign.value, state, static_tools, custom_tools)?;
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
                                            Box::new(CustomConstant::from(values[i].clone())),
                                        );
                                    }
                                    _ => panic!("Expected string"),
                                }
                            }
                        }
                        _ => panic!("Expected string"),
                    }
                }
                println!("State: a{:?}", state["a"].downcast_ref::<CustomConstant>());
                println!("State: b{:?}", state["b"].downcast_ref::<CustomConstant>());
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
        Box<dyn Fn(Vec<Constant>) -> Result<CustomConstant, InterpreterError>>,
    >,
    custom_tools: &HashMap<String, ToolFunction>,
) -> Result<CustomConstant, InterpreterError> {
    match &**expr {
        ast::Expr::Call(call) => {
            let func = match &*call.func {
                ast::Expr::Name(name) => name.id.to_string(),
                ast::Expr::Attribute(attr) => {
                    let func = evaluate_expr(
                        &Box::new(*attr.value.clone()),
                        state,
                        static_tools,
                        custom_tools,
                    )?;
                    let func = func.str().unwrap();
                    func
                }
                _ => panic!("Expected function name"),
            };
            let args = call
                .args
                .iter()
                .map(|e| evaluate_expr(&Box::new(e.clone()), state, static_tools, custom_tools))
                .collect::<Result<Vec<CustomConstant>, InterpreterError>>()?;
            println!("Function: {:?}", func);
            println!("Args: {:?}", args);
            if static_tools.contains_key(&func) {
                println!("Static tool");
                let result = static_tools[&func](args.iter().map(|c| Constant::from(c.clone())).collect());
                match result.as_ref() {
                    Ok(result) => match result {
                        CustomConstant::Str(s) => {
                            println!("Result: {}", s);
                        }
                        _ => {
                            println!("Result: {:?}", result);
                        }
                    },
                    Err(e) => {
                        println!("Error: {:?}", e);
                    }
                }
                result
            } else if custom_tools.contains_key(&func) {
                println!("Custom tool");
                let result = custom_tools[&func](args.iter().map(|c| Constant::from(c.clone())).collect());
                match result.as_ref() {
                    Ok(result) => match result {
                        CustomConstant::Str(s) => {
                            println!("Result: {}", s);
                        }
                        _ => {
                            println!("Result: {:?}", result);
                        }
                    },
                    Err(e) => {
                        println!("Error: {:?}", e);
                    }
                }
                result
            } else {
                Err(InterpreterError::RuntimeError(format!(
                    "Function '{}' not found",
                    func
                )))
            }
        }
        ast::Expr::BinOp(binop) => {
            let left_val = evaluate_expr(&binop.left.clone(), state, static_tools, custom_tools)?
                .float()
                .unwrap();
            let right_val = evaluate_expr(&binop.right.clone(), state, static_tools, custom_tools)?
                .float()
                .unwrap();
            match &binop.op {
                Operator::Add => {
                    println!("{} + {} = {}", left_val, right_val, left_val + right_val);
                    Ok(CustomConstant::Float(left_val + right_val))
                }
                Operator::Sub => {
                    println!("{} - {} = {}", left_val, right_val, left_val - right_val);
                    Ok(CustomConstant::Float(left_val - right_val))
                }
                Operator::Mult => {
                    println!("{} * {} = {}", left_val, right_val, left_val * right_val);
                    Ok(CustomConstant::Float(left_val * right_val))
                }
                Operator::Div => {
                    println!("{} / {} = {}", left_val, right_val, left_val / right_val);
                    Ok(CustomConstant::Float(left_val / right_val))
                }
                Operator::FloorDiv => {
                    println!("{} // {} = {}", left_val, right_val, left_val / right_val);
                    Ok(CustomConstant::Float(left_val / right_val))
                }
                Operator::Mod => {
                    println!("{} % {} = {}", left_val, right_val, left_val % right_val);
                    Ok(CustomConstant::Float(left_val % right_val))
                }
                Operator::Pow => {
                    println!(
                        "{} ** {} = {}",
                        left_val,
                        right_val,
                        left_val.powf(right_val)
                    );
                    Ok(CustomConstant::Float(left_val.powf(right_val)))
                }
                Operator::BitOr => {
                    println!("{}", 1 | 0);

                    println!(
                        "{} | {} = {}",
                        left_val,
                        right_val,
                        left_val as i64 | right_val as i64
                    );
                    Ok(CustomConstant::Int(BigInt::from(
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
                    Ok(CustomConstant::Int(BigInt::from(
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
                    Ok(CustomConstant::Int(BigInt::from(
                        left_val as i64 & right_val as i64,
                    )))
                }
                Operator::LShift => {
                    let left_val = left_val as i64;
                    let right_val = right_val as i64;
                    println!("{} << {} = {}", left_val, right_val, left_val << right_val);
                    Ok(CustomConstant::Int(BigInt::from(left_val << right_val)))
                }
                Operator::RShift => {
                    let left_val = left_val as i64;
                    let right_val = right_val as i64;
                    println!("{} >> {} = {}", left_val, right_val, left_val >> right_val);
                    Ok(CustomConstant::Int(BigInt::from(left_val >> right_val)))
                }
                Operator::MatMult => {
                    println!("{} * {} = {}", left_val, right_val, left_val * right_val);
                    Ok(CustomConstant::Float(left_val * right_val))
                }
            }
        }
        ast::Expr::UnaryOp(unaryop) => {
            let operand = evaluate_expr(&unaryop.operand, state, static_tools, custom_tools)?;
            match &unaryop.op {
                UnaryOp::USub => {
                    if let CustomConstant::Float(f) = operand {
                        Ok(CustomConstant::Float(-f))
                    } else {
                        panic!("Expected float")
                    }
                }
                UnaryOp::UAdd => Ok(operand),
                UnaryOp::Not => {
                    if let CustomConstant::Bool(b) = operand {
                        Ok(CustomConstant::Bool(!b))
                    } else {
                        panic!("Expected boolean")
                    }
                }
                UnaryOp::Invert => {
                    if let CustomConstant::Float(f) = operand {
                        Ok(CustomConstant::Float(-(f as i64) as f64))
                    } else {
                        panic!("Expected float")
                    }
                }
            }
        }
        ast::Expr::Constant(constant) => match &constant.value {
            Constant::Int(i) => Ok(CustomConstant::Float(convert_bigint_to_f64(&i))),
            _ => Ok(constant.value.clone().into()),
        },
        ast::Expr::List(list) => Ok(CustomConstant::Tuple(
            list.elts
                .iter()
                .map(|e| Constant::from(evaluate_expr(&Box::new(e.clone()), state, static_tools, custom_tools).unwrap()))
                .collect::<Vec<Constant>>(),
        )),
        ast::Expr::Name(name) => {
            if state.contains_key(&name.id.to_string()) {
                Ok(state[&name.id.to_string()]
                    .downcast_ref::<CustomConstant>()
                    .unwrap()
                    .clone())
            } else {
                Err(InterpreterError::RuntimeError(format!(
                    "Variable '{}' used before assignment",
                    name.id
                )))
            }
        }
        ast::Expr::Tuple(tuple) => Ok(CustomConstant::Tuple(
            tuple
                .elts
                .iter()
                .map(|e| Constant::from(evaluate_expr(&Box::new(e.clone()), state, static_tools, custom_tools).unwrap()))
                .collect::<Vec<Constant>>(),
        )),
        _ => {
            panic!("Unsupported expression: {:?}", expr);
        }
    }
}
