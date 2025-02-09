use crate::tools::AnyTool;
use anyhow::Result;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule, PyTuple};
use rustpython_parser::ast::ExprConstant;
use rustpython_parser::{
    ast::{
        self,
        bigint::{BigInt, Sign},
        Constant, Expr, Operator, Stmt, UnaryOp,
    },
    Parse,
};
use serde_json::{self, json};
use std::fmt;
use std::{any::Any, collections::HashMap};

pub fn get_base_python_tools() -> HashMap<&'static str, &'static str> {
    [
        ("print", "custom_print"),
        ("isinstance", "isinstance"),
        ("range", "range"),
        ("float", "float"),
        ("int", "int"),
        ("bool", "bool"),
        ("str", "str"),
        ("set", "set"),
        ("list", "list"),
        ("dict", "dict"),
        ("tuple", "tuple"),
        ("round", "round"),
        ("ceil", "math.ceil"),
        ("floor", "math.floor"),
        ("log", "math.log"),
        ("exp", "math.exp"),
        ("sin", "math.sin"),
        ("cos", "math.cos"),
        ("tan", "math.tan"),
        ("asin", "math.asin"),
        ("acos", "math.acos"),
        ("atan", "math.atan"),
        ("atan2", "math.atan2"),
        ("degrees", "math.degrees"),
        ("radians", "math.radians"),
        ("pow", "math.pow"),
        ("sqrt", "math.sqrt"),
        ("len", "len"),
        ("sum", "sum"),
        ("max", "max"),
        ("min", "min"),
        ("abs", "abs"),
        ("enumerate", "enumerate"),
        ("zip", "zip"),
        ("reversed", "reversed"),
        ("sorted", "sorted"),
        ("all", "all"),
        ("any", "any"),
        ("map", "map"),
        ("filter", "filter"),
        ("ord", "ord"),
        ("chr", "chr"),
        ("next", "next"),
        ("iter", "iter"),
        ("divmod", "divmod"),
        ("callable", "callable"),
        ("getattr", "getattr"),
        ("hasattr", "hasattr"),
        ("setattr", "setattr"),
        ("issubclass", "issubclass"),
        ("type", "type"),
        ("complex", "complex"),
    ]
    .iter()
    .cloned()
    .collect()
}

// Custom error type for interpreter
#[derive(Debug, PartialEq)]
pub enum InterpreterError {
    SyntaxError(String),
    RuntimeError(String),
    FinalAnswer(String),
    OperationLimitExceeded,
    UnauthorizedImport(String),
    UnsupportedOperation(String),
}

impl fmt::Display for InterpreterError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            InterpreterError::SyntaxError(msg) => write!(f, "Syntax Error: {}", msg),
            InterpreterError::RuntimeError(msg) => write!(f, "Runtime Error: {}", msg),
            InterpreterError::FinalAnswer(msg) => write!(f, "Final Answer: {}", msg),
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
    PyObj(PyObject),
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
            CustomConstant::Float(f) => Some(f.to_string()),
            CustomConstant::Int(i) => Some(i.to_string()),
            CustomConstant::Tuple(t) => {
                let mut result = String::new();
                result.push('[');

                for (i, item) in t.iter().enumerate() {
                    if i > 0 {
                        result.push_str(", ");
                    }
                    result.push_str(&CustomConstant::from(item.clone()).str().unwrap_or_default());
                }
                result.push(']');
                println!("Tuple result: {:?}", result);
                Some(result)
            }
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

impl IntoPy<PyObject> for CustomConstant {
    fn into_py(self, py: Python<'_>) -> PyObject {
        match self {
            CustomConstant::Int(i) => i.to_string().into_py(py),
            CustomConstant::Float(f) => f.into_py(py),
            CustomConstant::Str(s) => s.into_py(py),
            CustomConstant::Bool(b) => b.into_py(py),
            CustomConstant::Tuple(t) => {
                let py_list: Vec<f64> = t
                    .iter()
                    .map(|x| match x {
                        Constant::Float(f) => *f,
                        Constant::Int(i) => convert_bigint_to_f64(i),
                        _ => 0.0,
                    })
                    .collect();
                py_list.into_py(py)
            }
            CustomConstant::PyObj(obj) => obj,
        }
    }
}

type ToolFunction = Box<dyn Fn(Vec<Constant>) -> Result<CustomConstant, InterpreterError>>;
type CustomToolFunction =
    Box<dyn Fn(Vec<Constant>, HashMap<String, String>) -> Result<CustomConstant, InterpreterError>>;

fn setup_custom_tools(tools: Vec<Box<dyn AnyTool>>) -> HashMap<String, CustomToolFunction> {
    let mut tools_map = HashMap::new();
    for tool in tools {
        let tool_info = tool.tool_info();
        tools_map.insert(
            tool.name().to_string(),
            Box::new(
                move |args: Vec<Constant>, kwargs: HashMap<String, String>| {
                    //merge args and kwargs
                    let tool_parameter_names = tool_info.get_parameter_names();

                    let mut new_args = HashMap::new();
                    for (i, arg) in args.iter().enumerate() {
                        new_args
                            .insert(tool_parameter_names[i].clone(), arg.clone().str().unwrap());
                    }
                    for (key, value) in kwargs {
                        new_args.insert(key, value);
                    }
                    match tool.forward_json(json!(new_args)) {
                        Ok(results) => Ok(CustomConstant::Str(results)),
                        Err(e) => Ok(CustomConstant::Str(format!("Error: {}", e))),
                    }
                },
            ) as CustomToolFunction,
        );
    }
    tools_map
}

pub fn setup_static_tools(
    static_tools: HashMap<&'static str, &'static str>,
) -> HashMap<String, ToolFunction> {
    let mut tools = HashMap::new();
    let static_tools_clone = static_tools.clone();
    let eval_py = move |func: &str, args: Vec<Constant>| {
        Python::with_gil(|py| {
            let locals = PyDict::new(py);

            // Import required modules
            let math = PyModule::import(py, "math")?;
            locals.set_item("math", math)?;

            for (i, arg) in args.iter().enumerate() {
                match arg {
                    Constant::Float(f) => locals.set_item(format!("arg{}", i), f)?,
                    Constant::Int(int) => {
                        locals.set_item(format!("arg{}", i), convert_bigint_to_i64(int))?;
                    }
                    Constant::Str(s) => locals.set_item(format!("arg{}", i), s)?,
                    Constant::Tuple(t) => {
                        let py_list: Vec<f64> = t
                            .iter()
                            .map(|x| match x {
                                Constant::Float(f) => *f,
                                Constant::Int(i) => convert_bigint_to_f64(i),
                                _ => 0.0,
                            })
                            .collect();
                        locals.set_item(format!("arg{}", i), py_list)?
                    }
                    _ => locals.set_item(format!("arg{}", i), 0.0)?,
                }
            }

            let arg_names: Vec<String> = (0..args.len()).map(|i| format!("arg{}", i)).collect();
            let func_path = static_tools.get(func).unwrap_or(&"builtins.float");
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
            } else if let Ok(int_val) = result.extract::<i64>() {
                Ok(CustomConstant::Int(BigInt::from(int_val)))
            } else {
                Ok(CustomConstant::PyObj(result.into_py(py)))
            }
        })
    };

    // Register tools after eval_py is defined
    for func in static_tools_clone.keys() {
        let func = func.to_string(); // Create owned String
        let eval_py = eval_py.clone(); // Clone the closure
        tools.insert(
            func.clone(),
            Box::new(move |args| eval_py(&func, args)) as ToolFunction,
        );
    }

    tools
}

fn evaluate_ast(
    ast: &ast::Suite,
    state: &mut HashMap<String, Box<dyn Any>>,
    static_tools: &HashMap<
        String,
        Box<dyn Fn(Vec<Constant>) -> Result<CustomConstant, InterpreterError>>,
    >,
    custom_tools: &HashMap<String, CustomToolFunction>,
) -> Result<CustomConstant, InterpreterError> {
    for node in ast.iter() {
        match node {
            Stmt::FunctionDef(func) => {
                println!("Function: {:?}", func.name);
                return Ok(CustomConstant::Str(format!("Function: {:?}", func.name)));
            }
            Stmt::Expr(expr) => {
                let result = evaluate_expr(&expr.value, state, static_tools, custom_tools)?;
                return Ok(result);
            }
            Stmt::For(for_stmt) => {
                println!("For: {:?}", for_stmt.iter);
                let iter =
                    evaluate_expr(&for_stmt.iter.clone(), state, static_tools, custom_tools)?;
                println!("Iter: {:?}", iter);
                // Convert PyObj iterator into a vector of values
                let values = match iter {
                    CustomConstant::PyObj(obj) => {
                        Python::with_gil(|py| -> Result<Vec<Constant>, InterpreterError> {
                            let iter = obj.as_ref(py).iter()?;
                            let mut values = Vec::new();

                            for item in iter {
                                let item = item?;
                                if let Ok(num) = item.extract::<i64>() {
                                    values.push(Constant::Int(BigInt::from(num)));
                                } else if let Ok(float) = item.extract::<f64>() {
                                    values.push(Constant::Float(float));
                                } else if let Ok(string) = item.extract::<String>() {
                                    values.push(Constant::Str(string));
                                } else {
                                    return Err(InterpreterError::RuntimeError(
                                        "Unsupported type in iterator".to_string(),
                                    ));
                                }
                            }
                            Ok(values)
                        })?
                    }
                    CustomConstant::Tuple(items) => items,
                    _ => {
                        return Err(InterpreterError::RuntimeError(
                            "Expected iterable".to_string(),
                        ))
                    }
                };

                // Get the target variable name
                let target_name = match &*for_stmt.target {
                    ast::Expr::Name(name) => name.id.to_string(),
                    _ => {
                        return Err(InterpreterError::RuntimeError(
                            "Expected name as loop target".to_string(),
                        ))
                    }
                };
                let mut for_loop_result = CustomConstant::Str(String::new());
                // Iterate over the values and execute the body for each iteration
                for value in values {
                    // Update the loop variable in the state
                    state.insert(target_name.clone(), Box::new(CustomConstant::from(value)));

                    // Execute each statement in the loop body
                    for stmt in &for_stmt.body {
                        match stmt {
                            Stmt::Expr(expr) => {
                                for_loop_result =
                                    evaluate_expr(&expr.value, state, static_tools, custom_tools)?;
                            }
                            // Add other statement types as needed
                            _ => {
                                return Err(InterpreterError::UnsupportedOperation(
                                    "Unsupported statement in for loop".to_string(),
                                ))
                            }
                        }
                    }
                }
                return Ok(for_loop_result);
            }

            Stmt::Assign(assign) => {
                for target in assign.targets.iter() {
                    // let target = evaluate_expr(&Box::new(target.clone()), state, static_tools)?;
                    match target {
                        ast::Expr::Name(name) => {
                            let value =
                                evaluate_expr(&assign.value, state, static_tools, custom_tools)?;
                            state
                                .insert(name.id.to_string(), Box::new(CustomConstant::from(value)));
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
            }

            _ => {}
        }
    }
    Ok(CustomConstant::Str(String::new()))
}

fn convert_bigint_to_f64(i: &BigInt) -> f64 {
    let i = i.to_u32_digits();
    let num = i.1.iter().fold(0i64, |acc, &d| acc * (1 << 32) + d as i64);
    match i.0 {
        Sign::Minus => -num as f64,
        Sign::NoSign | Sign::Plus => num as f64,
    }
}
fn convert_bigint_to_i64(i: &BigInt) -> i64 {
    let i = i.to_u32_digits();
    let num = i.1.iter().fold(0i64, |acc, &d| acc * (1 << 32) + d as i64);
    match i.0 {
        Sign::Minus => -num,
        Sign::NoSign | Sign::Plus => num,
    }
}

fn evaluate_expr(
    expr: &Box<Expr>,
    state: &mut HashMap<String, Box<dyn Any>>,
    static_tools: &HashMap<
        String,
        Box<dyn Fn(Vec<Constant>) -> Result<CustomConstant, InterpreterError>>,
    >,
    custom_tools: &HashMap<String, CustomToolFunction>,
) -> Result<CustomConstant, InterpreterError> {
    match &**expr {
        ast::Expr::Call(call) => {
            let args = call
                .args
                .iter()
                .map(|e| evaluate_expr(&Box::new(e.clone()), state, static_tools, custom_tools))
                .collect::<Result<Vec<CustomConstant>, InterpreterError>>()?;
            let func = match &*call.func {
                ast::Expr::Name(name) => name.id.to_string(),
                ast::Expr::Attribute(attr) => {
                    let obj = evaluate_expr(
                        &Box::new(*attr.value.clone()),
                        state,
                        static_tools,
                        custom_tools,
                    )?;
                    let func_name = attr.attr.to_string();
                    let output = Python::with_gil(|py| {
                        let obj = obj.into_py(py);
                        let func = obj.getattr(py, func_name.as_str()).unwrap();
                        let py_args = args
                            .iter()
                            .map(|a| a.clone().into_py(py))
                            .collect::<Vec<PyObject>>();
                        let py_tuple = PyTuple::new(py, py_args);
                        let result = func.call1(py, py_tuple).unwrap();

                        // Handle different return types
                        if let Ok(float_val) = result.extract::<f64>(py) {
                            CustomConstant::Float(float_val)
                        } else if let Ok(string_val) = result.extract::<String>(py) {
                            CustomConstant::Str(string_val)
                        } else if let Ok(bool_val) = result.extract::<bool>(py) {
                            CustomConstant::Bool(bool_val)
                        } else if let Ok(int_val) = result.extract::<i64>(py) {
                            CustomConstant::Int(BigInt::from(int_val))
                        } else {
                            CustomConstant::PyObj(result.into_py(py))
                        }
                    });
                    return Ok(output);
                }
                _ => panic!("Expected function name"),
            };

            let keywords = call
                .keywords
                .iter()
                .map(|k| {
                    let value = evaluate_expr(
                        &Box::new(k.value.clone()),
                        state,
                        static_tools,
                        custom_tools,
                    )?;
                    Ok((k.arg.as_ref().unwrap().to_string(), value.str().unwrap()))
                })
                .collect::<Result<HashMap<String, String>, InterpreterError>>()?;
            println!("Function: {:?}", func);
            println!("Args: {:?}", args);
            println!("Keywords: {:?}", keywords);
            if func == "final_answer" {
                if let Some(answer) = keywords.get("answer") {
                    return Err(InterpreterError::FinalAnswer(answer.to_string()));
                } else {
                    return Err(InterpreterError::FinalAnswer(
                        args.iter()
                            .map(|c| c.str().unwrap())
                            .collect::<Vec<String>>()
                            .join(" "),
                    ));
                }
            }
            if func == "print" {
                return Ok(CustomConstant::Str(
                    args.iter()
                        .map(|c| c.str().unwrap())
                        .collect::<Vec<String>>()
                        .join(" "),
                ));
            }
            if static_tools.contains_key(&func) {
                println!("Static tool");
                let result =
                    static_tools[&func](args.iter().map(|c| Constant::from(c.clone())).collect());
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
                let result = custom_tools[&func](
                    args.iter().map(|c| Constant::from(c.clone())).collect(),
                    keywords,
                );
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
            let left_val = evaluate_expr(&binop.left.clone(), state, static_tools, custom_tools)?;
            let left_val = match left_val {
                CustomConstant::Float(f) => f,
                CustomConstant::Int(i) => convert_bigint_to_f64(&i),
                _ => panic!("Expected float or int"),
            };
            let right_val = evaluate_expr(&binop.right.clone(), state, static_tools, custom_tools)?;
            let right_val = match right_val {
                CustomConstant::Float(f) => f,
                CustomConstant::Int(i) => convert_bigint_to_f64(&i),
                _ => panic!("Expected float or int"),
            };

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
                UnaryOp::USub => match operand {
                    CustomConstant::Float(f) => Ok(CustomConstant::Float(-f)),
                    CustomConstant::Int(i) => Ok(CustomConstant::Int(-i)),
                    _ => panic!("Expected float or int"),
                },
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
            Constant::Int(i) => Ok(CustomConstant::Int(i.clone())),
            _ => Ok(constant.value.clone().into()),
        },
        ast::Expr::List(list) => Ok(CustomConstant::Tuple(
            list.elts
                .iter()
                .map(|e| {
                    Constant::from(
                        evaluate_expr(&Box::new(e.clone()), state, static_tools, custom_tools)
                            .unwrap(),
                    )
                })
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
                .map(|e| {
                    Constant::from(
                        evaluate_expr(&Box::new(e.clone()), state, static_tools, custom_tools)
                            .unwrap(),
                    )
                })
                .collect::<Vec<Constant>>(),
        )),
        ast::Expr::JoinedStr(joinedstr) => Ok(CustomConstant::Str(
            joinedstr
                .values
                .iter()
                .map(|e| {
                    evaluate_expr(&Box::new(e.clone()), state, static_tools, custom_tools)
                        .unwrap()
                        .str()
                        .unwrap()
                })
                .collect::<Vec<String>>()
                .join(""),
        )),
        ast::Expr::FormattedValue(formattedvalue) => Ok(CustomConstant::Str(
            evaluate_expr(&formattedvalue.value, state, static_tools, custom_tools)
                .unwrap()
                .str()
                .unwrap(),
        )),
        ast::Expr::Subscript(subscript) => {
            if let ast::Expr::Slice(constant) = &*subscript.slice {
                let slice_values = evaluate_expr(&subscript.slice, state, static_tools, custom_tools)?;
                let (start, end, step) = match slice_values.clone() {
                    CustomConstant::Tuple(t) => {
                        let start = convert_bigint_to_i64(&t[0].clone().int().unwrap());
                        let end = convert_bigint_to_i64(&t[1].clone().int().unwrap());
                        let step = convert_bigint_to_i64(&t[2].clone().int().unwrap());
                        (start, end, step)
                    }
                    _ => panic!("Expected tuple"),
                };
                let value = match slice_values {
                    CustomConstant::Tuple(t) => {
                        let value = evaluate_expr(&subscript.value, state, static_tools, custom_tools)?;
                        let value = match value {
                            CustomConstant::Tuple(t) => {
                                if step < 0 {
                                    t.iter()
                                        .rev()
                                        .skip((t.len() - start as usize - 1) as usize)
                                        .take((start - end) as usize)
                                        .step_by((-step) as usize)
                                        .map(|c| c.clone())
                                        .collect::<Vec<Constant>>()
                                } else {
                                    t.iter()
                                        .skip(start as usize)
                                        .take((end - start) as usize)
                                        .step_by(step as usize)
                                        .map(|c| c.clone())
                                        .collect::<Vec<Constant>>()
                                }
                            }
                            _ => panic!("Expected tuple"),
                        };
                        value
                    }
                    CustomConstant::Str(s) => {
                        if step < 0 {
                            s.chars()
                                .rev()
                                .skip((s.chars().count() - start as usize - 1) as usize)
                                .take((start - end) as usize)
                                .step_by((-step) as usize)
                                .map(|c| Constant::Str(c.to_string()))
                                .collect::<Vec<Constant>>()
                        } else {
                            s.chars()
                                .skip(start as usize)
                                .take((end - start) as usize)
                                .step_by(step as usize)
                                .map(|c| Constant::Str(c.to_string()))
                                .collect::<Vec<Constant>>()
                        }
                    }
                    _ => panic!("Expected tuple or string"),
                };
                return Ok(CustomConstant::Tuple(value));
            }

            let index = evaluate_expr(&subscript.slice, state, static_tools, custom_tools)?;
            let index = match index {
                CustomConstant::Int(i) => i,
                _ => panic!("Expected int"),
            };
            let value = evaluate_expr(&subscript.value, state, static_tools, custom_tools)?;
            let value = match value {
                CustomConstant::Tuple(t) => {
                    let index = convert_bigint_to_i64(&index);
                    if index >= t.len() as i64 || -index > t.len() as i64 {
                        return Err(InterpreterError::RuntimeError(format!(
                            "Index out of bounds: {}. There are only {} elements in the tuple.",
                            index,
                            t.len()
                        )));
                    }
                    if index < 0 {
                        t.iter().rev().nth(-index as usize - 1).unwrap().clone()
                    } else {
                        t[index as usize].clone()
                    }
                }
                CustomConstant::Str(s) => {
                    let index = convert_bigint_to_i64(&index);
                    let length = s.chars().count();
                    if index >= length as i64 || -index > length as i64 {
                        return Err(InterpreterError::RuntimeError(format!(
                            "Index out of bounds: {}. There are only {} characters in the string.",
                            index, length
                        )));
                    }
                    if index < 0 {
                        Constant::Str(
                            s.chars()
                                .rev()
                                .nth(-index as usize - 1)
                                .unwrap()
                                .to_string(),
                        )
                    } else {
                        Constant::Str(s.chars().nth(index as usize).unwrap().to_string())
                    }
                }
                _ => panic!("Expected tuple or string"),
            };
            Ok(CustomConstant::from(value))
        }
        ast::Expr::Slice(slice) => {
            let start = match &slice.lower {
                Some(lower) => evaluate_expr(&lower, state, static_tools, custom_tools)?.into(),
                None => Constant::Int(BigInt::from(0)),
            };
            let end = match &slice.upper {
                Some(upper) => evaluate_expr(&upper, state, static_tools, custom_tools)?.into(),
                None => Constant::Int(BigInt::from(0)),
            };
            let step = match &slice.step {
                Some(step) => evaluate_expr(&step, state, static_tools, custom_tools)?.into(),
                None => Constant::Int(BigInt::from(1)),
            };
            // let start = match start {
            //     CustomConstant::Int(i) => convert_bigint_to_i64(&i),
            //     _ => panic!("Expected int"),
            // };
            // let end = match end {
            //     CustomConstant::Int(i) => convert_bigint_to_i64(&i),
            //     _ => panic!("Expected int"),
            // };
            // let step = match step {
            //     CustomConstant::Int(i) => convert_bigint_to_i64(&i),
            //     _ => panic!("Expected int"),
            // };
            Ok(CustomConstant::Tuple(vec![start, end, step]))
        }
        _ => {
            panic!("Unsupported expression: {:?}", expr);
        }
    }
}

pub fn evaluate_python_code(
    code: &str,
    custom_tools: Vec<Box<dyn AnyTool>>,
    state: &mut HashMap<String, Box<dyn Any>>,
) -> Result<String, InterpreterError> {
    let base_tools = get_base_python_tools();
    let static_tools = setup_static_tools(base_tools);
    let custom_tools = setup_custom_tools(custom_tools);
    let ast = ast::Suite::parse(code, "<embedded>")
        .map_err(|e| InterpreterError::SyntaxError(e.to_string()))?;

    let result = evaluate_ast(&ast, state, &static_tools, &custom_tools)?;
    Ok(result.str().unwrap())
}

pub struct LocalPythonInterpreter {
    static_tools: HashMap<String, ToolFunction>,
    custom_tools: HashMap<String, CustomToolFunction>,
}

impl LocalPythonInterpreter {
    pub fn new(custom_tools: Vec<Box<dyn AnyTool>>) -> Self {
        let custom_tools = setup_custom_tools(custom_tools);
        let base_tools = get_base_python_tools();
        let static_tools = setup_static_tools(base_tools);
        Self {
            static_tools,
            custom_tools,
        }
    }
    pub fn forward(
        &self,
        code: &str,
        state: &mut Option<HashMap<String, Box<dyn Any>>>,
    ) -> Result<String, InterpreterError> {
        let ast = ast::Suite::parse(code, "<embedded>")
            .map_err(|e| InterpreterError::SyntaxError(e.to_string()))?;
        println!("Tools: {:?}", self.custom_tools.keys());
        let result = evaluate_ast(
            &ast,
            state.as_mut().unwrap_or(&mut HashMap::new()),
            &self.static_tools,
            &self.custom_tools,
        )?;
        match result.str() {
            Some(s) => Ok(s),
            None => Err(InterpreterError::RuntimeError("No result".to_string())),
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::FinalAnswerTool;
    use std::collections::HashMap;

    #[test]
    fn test_evaluate_python_code() {
        let code = "print('Hello, world!')";
        let mut state = HashMap::new();
        let result = evaluate_python_code(code, vec![], &mut state).unwrap();
        assert_eq!(result, "Hello, world!");
    }

    #[test]
    fn test_evaluate_python_code_with_joined_str() {
        let code = r#"word = 'strawberry'
r_count = word.count('r')
print(f"The letter 'r' appears {r_count} times in the word '{word}'.")"#;
        let mut state = HashMap::new();
        let result = evaluate_python_code(code, vec![], &mut state).unwrap();
        assert_eq!(
            result,
            "The letter 'r' appears 3 times in the word 'strawberry'."
        );
    }

    #[test]
    fn test_final_answer_execution() {
        let tools: Vec<Box<dyn AnyTool>> = vec![Box::new(FinalAnswerTool::new())];
        let mut state = HashMap::new();
        let result =
            evaluate_python_code("final_answer(answer='Hello, world!')", tools, &mut state);
        assert_eq!(
            result,
            Err(InterpreterError::FinalAnswer("Hello, world!".to_string()))
        );
    }

    #[test]
    fn test_evaluate_python_code_with_subscript() {
        let code = textwrap::dedent(
            r#"
        word = 'strawberry'
        print(word[3])"#,
        );
        println!("Code: {}", code);
        let mut state = HashMap::new();
        let result = evaluate_python_code(&code, vec![], &mut state).unwrap();
        assert_eq!(result, "a");

        let code = textwrap::dedent(
            r#"
        word = 'strawberry'
        print(word[-3])"#,
        );
        let mut state = HashMap::new();
        let result = evaluate_python_code(&code, vec![], &mut state).unwrap();
        assert_eq!(result, "r");

        let code = textwrap::dedent(
            r#"
        word = 'strawberry'
        print(word[9])"#,
        );
        let mut state = HashMap::new();
        let result = evaluate_python_code(&code, vec![], &mut state).unwrap();
        assert_eq!(result, "y");

        let code = textwrap::dedent(
            r#"
        word = 'strawberry'
        print(word[10])"#,
        );
        let mut state = HashMap::new();
        let result = evaluate_python_code(&code, vec![], &mut state);
        assert_eq!(
            result,
            Err(InterpreterError::RuntimeError(
                "Index out of bounds: 10. There are only 10 characters in the string.".to_string()
            ))
        );

        let code = textwrap::dedent(
            r#"
        numbers = [1, 2, 3, 4, 5]
        print(numbers[1])"#,
        );
        let mut state = HashMap::new();
        let result = evaluate_python_code(&code, vec![], &mut state).unwrap();
        assert_eq!(result, "2");

        let code = textwrap::dedent(
            r#"
        numbers = [1, 2, 3, 4, 5]
        print(numbers[-5])"#,
        );
        let mut state = HashMap::new();
        let result = evaluate_python_code(&code, vec![], &mut state).unwrap();
        assert_eq!(result, "1");

        let code = textwrap::dedent(
            r#"
        numbers = [1, 2, 3, 4, 5]
        print(numbers[-6])"#,
        );
        let mut state = HashMap::new();
        let result = evaluate_python_code(&code, vec![], &mut state);
        assert_eq!(
            result,
            Err(InterpreterError::RuntimeError(
                "Index out of bounds: -6. There are only 5 elements in the tuple.".to_string()
            ))
        );
    }

    #[test]
    fn test_evaluate_python_code_with_slice() {
        let code = textwrap::dedent(
            r#"
        numbers = [1, 2, 3, 4, 5]
        print(numbers[1:3])"#,
        );
        let mut state = HashMap::new();
        let result = evaluate_python_code(&code, vec![], &mut state).unwrap();
        assert_eq!(result, "[2, 3]");

        let code = textwrap::dedent(
            r#"
        numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        print(numbers[1:5:2])"#,
        );
        let mut state = HashMap::new();
        let result = evaluate_python_code(&code, vec![], &mut state).unwrap();
        assert_eq!(result, "[2, 4]");

        let code = textwrap::dedent(
            r#"
        numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        print(numbers[5:1:-2])"#,
        );
        let mut state = HashMap::new();
        let result = evaluate_python_code(&code, vec![], &mut state).unwrap();
        assert_eq!(result, "[6, 4]");
    }
}
