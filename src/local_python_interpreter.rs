use crate::errors::InterpreterError;
use crate::tools::AnyTool;
use anyhow::Result;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule, PyTuple};
use rustpython_parser::{
    ast::{
        self,
        bigint::{BigInt, Sign},
        Constant, Expr, Operator, Stmt, UnaryOp,
    },
    Parse,
};
use serde_json::{self, json};
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
    Tuple(Vec<CustomConstant>),
    PyObj(PyObject),
    Dict(Vec<String>, Vec<CustomConstant>),
}

impl CustomConstant {
    pub fn float(&self) -> Option<f64> {
        match self {
            CustomConstant::Float(f) => Some(*f),
            _ => None,
        }
    }
    pub fn str(&self) -> String {
        match self {
            CustomConstant::Str(s) => s.clone(),
            CustomConstant::Float(f) => f.to_string(),
            CustomConstant::Int(i) => i.to_string(),
            CustomConstant::Tuple(t) => {
                let mut result = String::new();
                result.push('[');
                for (i, item) in t.iter().enumerate() {
                    if i > 0 {
                        result.push_str(", ");
                    }
                    result.push_str(&item.str());
                }
                result.push(']');
                result
            }
            CustomConstant::Dict(keys, values) => {
                let mut result = String::new();
                result.push('{');
                for (i, key) in keys.iter().enumerate() {
                    if i > 0 {
                        result.push_str(", ");
                    }
                    result.push_str(&format!("'{}': {}", key, values[i].str()));
                }
                result.push('}');

                for (i, item) in values.iter().enumerate() {
                    if i > 0 {
                        result.push_str(", ");
                    }
                    result.push_str(&item.str());
                }
                result.push('}');
                result
            }
            CustomConstant::PyObj(obj) => obj.to_string(),
            CustomConstant::Bool(b) => b.to_string(),
        }
    }
    pub fn tuple(&self) -> Option<Vec<CustomConstant>> {
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
            CustomConstant::PyObj(obj) => Constant::Str(obj.to_string()),
            CustomConstant::Tuple(t) => {
                let tuple_items = t
                    .iter()
                    .map(|c| Constant::from(c.clone()))
                    .collect::<Vec<Constant>>();
                Constant::Tuple(tuple_items)
            }
            CustomConstant::Dict(keys, values) => {
                let tuple_items = keys
                    .iter()
                    .zip(values.iter())
                    .map(|(k, v)| {
                        Constant::Tuple(vec![Constant::Str(k.clone()), Constant::from(v.clone())])
                    })
                    .collect::<Vec<Constant>>();
                Constant::Tuple(tuple_items)
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
            Constant::None => CustomConstant::Str("None".to_string()),
            Constant::Tuple(t) => {
                CustomConstant::Tuple(t.iter().map(|c| c.clone().into()).collect())
            }
            _ => panic!("Unsupported constant type"),
        }
    }
}

impl IntoPy<PyObject> for CustomConstant {
    fn into_py(self, py: Python<'_>) -> PyObject {
        match self {
            CustomConstant::Int(i) => convert_bigint_to_i64(&i).into_py(py),
            CustomConstant::Float(f) => f.into_py(py),
            CustomConstant::Str(s) => s.into_py(py),
            CustomConstant::Bool(b) => b.into_py(py),
            CustomConstant::Tuple(t) => {
                let py_list = t
                    .iter()
                    .map(|x| x.clone().into_py(py))
                    .collect::<Vec<PyObject>>();
                py_list.into_py(py)
            }
            CustomConstant::PyObj(obj) => obj,
            CustomConstant::Dict(keys, values) => {
                let dict = PyDict::new(py);
                for (key, value) in keys.iter().zip(values.iter()) {
                    dict.set_item(key, value.clone().into_py(py))
                        .unwrap_or_default();
                }
                dict.into_py(py)
            }
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

            let result = py.eval(&expr, None, Some(locals))?;
            // Handle different return types
            if let Ok(float_val) = result.extract::<f64>() {
                Ok(CustomConstant::Float(float_val))
            } else if let Ok(list_val) = result.extract::<Vec<String>>() {
                Ok(CustomConstant::Tuple(
                    list_val.into_iter().map(CustomConstant::Str).collect(),
                ))
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

fn evaluate_stmt(
    node: &ast::Stmt,
    state: &mut HashMap<String, Box<dyn Any>>,
    static_tools: &HashMap<String, StaticTool>,
    custom_tools: &HashMap<String, CustomToolFunction>,
) -> Result<CustomConstant, InterpreterError> {
    match node {
        Stmt::FunctionDef(func) => Ok(CustomConstant::Str(format!("Function: {:?}", func.name))),
        Stmt::Expr(expr) => {
            let result = evaluate_expr(&expr.value, state, static_tools, custom_tools)?;
            Ok(result)
        }
        Stmt::For(for_stmt) => {
            let iter = evaluate_expr(&for_stmt.iter.clone(), state, static_tools, custom_tools)?;
            // Convert PyObj iterator into a vector of values
            let values = match iter {
                CustomConstant::PyObj(obj) => {
                    Python::with_gil(|py| -> Result<Vec<CustomConstant>, InterpreterError> {
                        let iter = obj.as_ref(py).iter()?;
                        let mut values = Vec::new();

                        for item in iter {
                            let item = item?;
                            if let Ok(num) = item.extract::<i64>() {
                                values.push(CustomConstant::Int(BigInt::from(num)));
                            } else if let Ok(float) = item.extract::<f64>() {
                                values.push(CustomConstant::Float(float));
                            } else if let Ok(string) = item.extract::<String>() {
                                values.push(CustomConstant::Str(string));
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
                state.insert(target_name.clone(), Box::new(value));

                // Execute each statement in the loop body
                for stmt in &for_stmt.body {
                    for_loop_result = evaluate_stmt(stmt, state, static_tools, custom_tools)?;
                }
            }
            Ok(for_loop_result)
        }

        Stmt::Assign(assign) => {
            for target in assign.targets.iter() {
                // let target = evaluate_expr(&Box::new(target.clone()), state, static_tools)?;
                match target {
                    ast::Expr::Name(name) => {
                        let value =
                            evaluate_expr(&assign.value, state, static_tools, custom_tools)?;
                        state.insert(name.id.to_string(), Box::new(value));
                    }
                    ast::Expr::Tuple(target_names) => {
                        let value =
                            evaluate_expr(&assign.value, state, static_tools, custom_tools)?;
                        let values = value.tuple().ok_or_else(|| {
                            InterpreterError::RuntimeError(
                                "Tuple unpacking failed. Expected values of type tuple".to_string(),
                            )
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
                                    state.insert(name.id.to_string(), Box::new(values[i].clone()));
                                }
                                _ => panic!("Expected string"),
                            }
                        }
                    }
                    _ => panic!("Expected string"),
                }
            }
            Ok(CustomConstant::Str(String::new()))
        }

        _ => Err(InterpreterError::RuntimeError(format!(
            "Unsupported statement {:?}",
            node
        ))),
    }
}

fn evaluate_ast(
    ast: &ast::Suite,
    state: &mut HashMap<String, Box<dyn Any>>,
    static_tools: &HashMap<String, StaticTool>,
    custom_tools: &HashMap<String, CustomToolFunction>,
) -> Result<CustomConstant, InterpreterError> {
    let mut result = CustomConstant::Str(String::new());
    for node in ast.iter() {
        result = evaluate_stmt(node, state, static_tools, custom_tools)?;
    }
    Ok(result)
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

type StaticTool = Box<dyn Fn(Vec<Constant>) -> Result<CustomConstant, InterpreterError>>;
type CustomTool =
    Box<dyn Fn(Vec<Constant>, HashMap<String, String>) -> Result<CustomConstant, InterpreterError>>;

fn evaluate_expr(
    expr: &Expr,
    state: &mut HashMap<String, Box<dyn Any>>,
    static_tools: &HashMap<String, StaticTool>,
    custom_tools: &HashMap<String, CustomTool>,
) -> Result<CustomConstant, InterpreterError> {
    match &expr {
        ast::Expr::Dict(dict) => {
            let keys = dict
                .keys
                .iter()
                .map(|e| {
                    evaluate_expr(
                        &Box::new(e.clone().ok_or_else(|| {
                            InterpreterError::RuntimeError(
                                "Dictionary key cannot be None".to_string(),
                            )
                        })?),
                        state,
                        static_tools,
                        custom_tools,
                    )
                    .map(|c| c.str())
                })
                .collect::<Result<Vec<String>, _>>()?;
            let values = dict
                .values
                .iter()
                .map(|e| evaluate_expr(&Box::new(e.clone()), state, static_tools, custom_tools))
                .collect::<Result<Vec<CustomConstant>, _>>()?;
            Ok(CustomConstant::Dict(keys, values))
        }
        ast::Expr::ListComp(list_comp) => {
            let iter = evaluate_expr(
                &list_comp.generators[0].iter,
                state,
                static_tools,
                custom_tools,
            )?;
            let result = Python::with_gil(|py| -> Result<Vec<CustomConstant>, InterpreterError> {
                let iter = iter.into_py(py);
                let iter = iter.as_ref(py).iter()?;
                let mut result = Vec::new();
                for item in iter {
                    let target = match &list_comp.generators[0].target {
                        ast::Expr::Name(name) => name.id.to_string(),
                        _ => panic!("Expected string"),
                    };
                    let item = item?;
                    let item = extract_constant_from_pyobject(item, py)?;
                    state.insert(target, Box::new(item));
                    let eval_expr =
                        evaluate_expr(&list_comp.elt, state, static_tools, custom_tools)?;
                    result.push(eval_expr);
                }
                Ok(result)
            });
            let result = result?;
            Ok(CustomConstant::Tuple(result))
        }
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
                    let output =
                        Python::with_gil(|py| -> Result<CustomConstant, InterpreterError> {
                            let obj = obj.into_py(py);
                            let func = obj.getattr(py, func_name.as_str())?;
                            let py_args = args
                                .iter()
                                .map(|a| match a {
                                    // Convert numeric types to strings when calling string methods
                                    CustomConstant::Float(f) => f.into_py(py),
                                    CustomConstant::Int(i) => convert_bigint_to_i64(i).into_py(py),
                                    _ => a.clone().into_py(py),
                                })
                                .collect::<Vec<PyObject>>();
                            let py_tuple = PyTuple::new(py, py_args);
                            let result = func.call1(py, py_tuple)?;

                            // For methods that modify in place (like append), return the original object
                            if func_name == "append"
                                || func_name == "extend"
                                || func_name == "insert"
                            {
                                let target = match &*attr.value {
                                    ast::Expr::Name(name) => name.id.to_string(),
                                    _ => panic!("Expected name"),
                                };
                                let out = extract_constant_from_pyobject(obj.as_ref(py), py)?;
                                state.insert(target, Box::new(out.clone()));
                                return Ok(out);
                            }

                            extract_constant_from_pyobject(result.as_ref(py), py)
                        });
                    return output;
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
                    Ok((k.arg.as_ref().unwrap().to_string(), value.str()))
                })
                .collect::<Result<HashMap<String, String>, InterpreterError>>()?;
            if func == "final_answer" {
                if let Some(answer) = keywords.get("answer") {
                    return Err(InterpreterError::FinalAnswer(answer.to_string()));
                } else {
                    return Err(InterpreterError::FinalAnswer(
                        args.iter()
                            .map(|c| c.str())
                            .collect::<Vec<String>>()
                            .join(" "),
                    ));
                }
            }
            if func == "print" {
                match state.get_mut("print_logs") {
                    Some(logs) => {
                        if let Some(logs) = logs.downcast_mut::<Vec<String>>() {
                            logs.push(
                                args.iter()
                                    .map(|c| c.str())
                                    .collect::<Vec<String>>()
                                    .join(" "),
                            );
                        } else {
                            return Err(InterpreterError::RuntimeError(
                                "print_logs is not a list".to_string(),
                            ));
                        }
                    }
                    None => {
                        state.insert(
                            "print_logs".to_string(),
                            Box::new(args.iter().map(|c| c.str()).collect::<Vec<String>>()),
                        );
                    }
                }
                return Ok(CustomConstant::Str(
                    args.iter()
                        .map(|c| c.str())
                        .collect::<Vec<String>>()
                        .join(" "),
                ));
            }
            if static_tools.contains_key(&func) {
                let result =
                    static_tools[&func](args.iter().map(|c| Constant::from(c.clone())).collect());
                result
            } else if custom_tools.contains_key(&func) {
                let result = custom_tools[&func](
                    args.iter().map(|c| Constant::from(c.clone())).collect(),
                    keywords,
                );
                result
            } else {
                Err(InterpreterError::RuntimeError(format!(
                    "Function '{}' not found",
                    func
                )))
            }
        }
        ast::Expr::BinOp(binop) => {
            let left_val_exp =
                evaluate_expr(&binop.left.clone(), state, static_tools, custom_tools)?;
            let right_val_exp: CustomConstant =
                evaluate_expr(&binop.right.clone(), state, static_tools, custom_tools)?;

            match binop.op {
                Operator::Add => match (left_val_exp.clone(), right_val_exp.clone()) {
                    (CustomConstant::Str(s), CustomConstant::Str(s2)) => {
                        return Ok(CustomConstant::Str(s + &s2));
                    }
                    (CustomConstant::Str(s), CustomConstant::Int(i)) => {
                        return Ok(CustomConstant::Str(s + &i.to_string()));
                    }
                    (CustomConstant::Int(i), CustomConstant::Str(s)) => {
                        return Ok(CustomConstant::Str(i.to_string() + &s));
                    }
                    _ => {}
                },
                Operator::Mult => match (left_val_exp.clone(), right_val_exp.clone()) {
                    (CustomConstant::Str(s), CustomConstant::Int(i)) => {
                        return Ok(CustomConstant::Str(
                            s.repeat(convert_bigint_to_i64(&i) as usize),
                        ));
                    }
                    (CustomConstant::Int(i), CustomConstant::Str(s)) => {
                        return Ok(CustomConstant::Str(
                            s.repeat(convert_bigint_to_i64(&i) as usize),
                        ));
                    }
                    _ => {}
                },
                _ => {}
            }
            let left_val = match left_val_exp.clone() {
                CustomConstant::Float(f) => f,
                CustomConstant::Int(i) => convert_bigint_to_f64(&i),
                _ => panic!("Expected float or int"),
            };
            let right_val = match right_val_exp.clone() {
                CustomConstant::Float(f) => f,
                CustomConstant::Int(i) => convert_bigint_to_f64(&i),
                _ => panic!("Expected float or int"),
            };

            match &binop.op {
                Operator::Add => Ok(CustomConstant::Float(left_val + right_val)),
                Operator::Sub => Ok(CustomConstant::Float(left_val - right_val)),
                Operator::Mult => Ok(CustomConstant::Float(left_val * right_val)),
                Operator::Div => Ok(CustomConstant::Float(left_val / right_val)),
                Operator::FloorDiv => Ok(CustomConstant::Float(left_val / right_val)),
                Operator::Mod => Ok(CustomConstant::Float(left_val % right_val)),
                Operator::Pow => Ok(CustomConstant::Float(left_val.powf(right_val))),
                Operator::BitOr => Ok(CustomConstant::Int(BigInt::from(
                    left_val as i64 | right_val as i64,
                ))),
                Operator::BitXor => Ok(CustomConstant::Int(BigInt::from(
                    left_val as i64 ^ right_val as i64,
                ))),
                Operator::BitAnd => Ok(CustomConstant::Int(BigInt::from(
                    left_val as i64 & right_val as i64,
                ))),
                Operator::LShift => {
                    let left_val = left_val as i64;
                    let right_val = right_val as i64;
                    Ok(CustomConstant::Int(BigInt::from(left_val << right_val)))
                }
                Operator::RShift => {
                    let left_val = left_val as i64;
                    let right_val = right_val as i64;
                    Ok(CustomConstant::Int(BigInt::from(left_val >> right_val)))
                }
                Operator::MatMult => Ok(CustomConstant::Float(left_val * right_val)),
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
                .map(|e| evaluate_expr(&Box::new(e.clone()), state, static_tools, custom_tools))
                .collect::<Result<Vec<CustomConstant>, _>>()?,
        )),
        ast::Expr::Name(name) => {
            if let Some(value) = state.get(name.id.as_str()) {
                if let Some(constant) = value.downcast_ref::<CustomConstant>() {
                    Ok(constant.clone())
                } else {
                    Err(InterpreterError::RuntimeError(format!(
                        "Error in downcasting constant {}",
                        name.id
                    )))
                }
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
                .map(|e| evaluate_expr(&Box::new(e.clone()), state, static_tools, custom_tools))
                .collect::<Result<Vec<CustomConstant>, _>>()?,
        )),
        ast::Expr::JoinedStr(joinedstr) => Ok(CustomConstant::Str(
            joinedstr
                .values
                .iter()
                .map(|e| {
                    evaluate_expr(&Box::new(e.clone()), state, static_tools, custom_tools)
                        .map(|result| result.str())
                })
                .collect::<Result<Vec<String>, _>>()?
                .join(""),
        )),
        ast::Expr::FormattedValue(formattedvalue) => {
            let result = evaluate_expr(&formattedvalue.value, state, static_tools, custom_tools)?;

            Ok(CustomConstant::Str(result.str()))
        }
        ast::Expr::Subscript(subscript) => {
            let result = Python::with_gil(|py| {
                // Get the value being subscripted (e.g., the list/string)
                let value = evaluate_expr(&subscript.value, state, static_tools, custom_tools)?;
                let value_obj = value.into_py(py);

                let slice = Constant::from(evaluate_expr(
                    &subscript.slice,
                    state,
                    static_tools,
                    custom_tools,
                )?);

                // Handle integer indices for lists/sequences
                if let Constant::Int(i) = slice {
                    let index = convert_bigint_to_i64(&i);
                    let result = value_obj.as_ref(py).get_item(index);
                    match result {
                        Ok(result) => return extract_constant_from_pyobject(result, py),
                        Err(e) => return Err(InterpreterError::RuntimeError(e.to_string())),
                    }
                }

                // Handle string keys for dictionaries
                if let Constant::Str(s) = slice {
                    // Try to extract as dictionary first
                    if let Ok(dict) = value_obj.as_ref(py).downcast::<PyDict>() {
                        let result = dict.get_item(s.clone());
                        match result {
                            Some(value) => return extract_constant_from_pyobject(value, py),
                            None => {
                                return Err(InterpreterError::RuntimeError(format!(
                                    "KeyError: '{}'",
                                    s
                                )))
                            }
                        }
                    }
                }

                // Handle both simple indexing and slicing
                let result = match &*subscript.slice {
                    // For slice operations like num[1:3:2]
                    ast::Expr::Slice(slice) => {
                        let start = match &slice.lower {
                            Some(lower) => {
                                evaluate_expr(lower, state, static_tools, custom_tools)?.into()
                            }
                            None => None,
                        };
                        let start = start
                            .map(|start| {
                                let constant = Constant::from(start);
                                constant
                                    .int()
                                    .map(|i| convert_bigint_to_i64(&i))
                                    .ok_or_else(|| {
                                        InterpreterError::RuntimeError(
                                            "Invalid start value in slice".to_string(),
                                        )
                                    })
                            })
                            .transpose()?;

                        let stop = match &slice.upper {
                            Some(upper) => {
                                evaluate_expr(upper, state, static_tools, custom_tools)?.into()
                            }
                            None => None,
                        };
                        let stop = stop
                            .map(|stop| {
                                let constant = Constant::from(stop);
                                constant
                                    .int()
                                    .map(|i| convert_bigint_to_i64(&i))
                                    .ok_or_else(|| {
                                        InterpreterError::RuntimeError(
                                            "Invalid stop value in slice".to_string(),
                                        )
                                    })
                            })
                            .transpose()?;

                        let step = match &slice.step {
                            Some(step) => {
                                evaluate_expr(step, state, static_tools, custom_tools)?.into()
                            }
                            None => None,
                        };
                        let step = step
                            .map(|step| {
                                let constant = Constant::from(step);
                                constant
                                    .int()
                                    .map(|i| convert_bigint_to_i64(&i))
                                    .ok_or_else(|| {
                                        InterpreterError::RuntimeError(
                                            "Invalid step value in slice".to_string(),
                                        )
                                    })
                            })
                            .transpose()?;

                        let slice_obj = py
                            .eval("slice", None, None)?
                            .call1((start, stop, step))?
                            .into_py(py);
                        value_obj.as_ref(py).get_item(slice_obj)?
                    }
                    _ => return Err(InterpreterError::RuntimeError("Invalid slice".to_string())),
                };

                // Convert the result back to our CustomConstant type
                extract_constant_from_pyobject(result, py)
            });
            result
        }
        ast::Expr::Slice(slice) => {
            let start = match &slice.lower {
                Some(lower) => evaluate_expr(lower, state, static_tools, custom_tools)?,
                None => CustomConstant::Int(BigInt::from(0)),
            };
            let end = match &slice.upper {
                Some(upper) => evaluate_expr(upper, state, static_tools, custom_tools)?,
                None => CustomConstant::Int(BigInt::from(0)),
            };
            let step = match &slice.step {
                Some(step) => evaluate_expr(step, state, static_tools, custom_tools)?,
                None => CustomConstant::Int(BigInt::from(1)),
            };
            Ok(CustomConstant::Tuple(vec![start, end, step]))
        }
        _ => {
            panic!("Unsupported expression: {:?}", expr);
        }
    }
}

fn extract_constant_from_pyobject(
    obj: &PyAny,
    py: Python<'_>,
) -> Result<CustomConstant, InterpreterError> {
    if let Ok(float_val) = obj.extract::<f64>() {
        Ok(CustomConstant::Float(float_val))
    } else if let Ok(string_val) = obj.extract::<String>() {
        Ok(CustomConstant::Str(string_val))
    } else if let Ok(bool_val) = obj.extract::<bool>() {
        Ok(CustomConstant::Bool(bool_val))
    } else if let Ok(int_val) = obj.extract::<i64>() {
        Ok(CustomConstant::Int(BigInt::from(int_val)))
    } else if let Ok(list_val) = obj.extract::<Vec<String>>() {
        Ok(CustomConstant::Tuple(
            list_val.into_iter().map(CustomConstant::Str).collect(),
        ))
    } else if let Ok(list_val) = obj.extract::<Vec<i64>>() {
        Ok(CustomConstant::Tuple(
            list_val
                .into_iter()
                .map(|i| CustomConstant::Int(BigInt::from(i)))
                .collect(),
        ))
    } else if let Ok(list_val) = obj.extract::<Vec<f64>>() {
        Ok(CustomConstant::Tuple(
            list_val.into_iter().map(CustomConstant::Float).collect(),
        ))
    } else if let Ok(dict_value) = obj.extract::<&PyDict>() {
        let keys = dict_value
            .keys()
            .iter()
            .map(|key| key.extract::<String>())
            .collect::<Result<Vec<String>, _>>()?;
        let values = dict_value
            .values()
            .iter()
            .map(|value| extract_constant_from_pyobject(value, py))
            .collect::<Result<Vec<CustomConstant>, _>>()?;
        Ok(CustomConstant::Dict(keys, values))
    } else {
        Ok(CustomConstant::PyObj(obj.into_py(py)))
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
    Ok(result.str())
}

pub struct LocalPythonInterpreter {
    static_tools: HashMap<String, ToolFunction>,
    custom_tools: HashMap<String, CustomToolFunction>,
    state: HashMap<String, Box<dyn Any>>,
}

impl LocalPythonInterpreter {
    pub fn new(custom_tools: Vec<Box<dyn AnyTool>>) -> Self {
        let custom_tools = setup_custom_tools(custom_tools);
        let base_tools = get_base_python_tools();
        let static_tools = setup_static_tools(base_tools);
        Self {
            static_tools,
            custom_tools,
            state: HashMap::new(),
        }
    }
    pub fn forward(
        &mut self,
        code: &str,
    ) -> Result<(String, String), InterpreterError> {
        let ast = ast::Suite::parse(code, "<embedded>")
            .map_err(|e| InterpreterError::SyntaxError(e.to_string()))?;
        let state = &mut self.state;
        let result = evaluate_ast(&ast, state, &self.static_tools, &self.custom_tools)?;

        let mut empty_string = Vec::new();
        let execution_logs = state
            .get_mut("print_logs")
            .and_then(|logs| logs.downcast_mut::<Vec<String>>())
            .unwrap_or(&mut empty_string)
            .join("\n");
        Ok((result.str(), execution_logs))
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::{DuckDuckGoSearchTool, FinalAnswerTool, VisitWebsiteTool};
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
                "IndexError: string index out of range".to_string()
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
                "IndexError: list index out of range".to_string()
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

        let code = textwrap::dedent(
            r#"
        word = 'strawberry'
        print(word[::-1])"#,
        );
        let mut state = HashMap::new();
        let result = evaluate_python_code(&code, vec![], &mut state).unwrap();
        assert_eq!(result, "yrrebwarts");

        let code = textwrap::dedent(
            r#"
        numbers = [1, 2, 3, 4, 5]
        print(numbers[::-1])"#,
        );
        let mut state = HashMap::new();
        let result = evaluate_python_code(&code, vec![], &mut state).unwrap();
        assert_eq!(result, "[5, 4, 3, 2, 1]");
    }

    #[test]
    fn test_for_loop() {
        let code = textwrap::dedent(
            r#"
        for i in range(5):
            print(i)
        "#,
        );
        let mut state = HashMap::new();
        let _ = evaluate_python_code(&code, vec![], &mut state).unwrap();
        assert_eq!(
            state
                .get("print_logs")
                .unwrap()
                .downcast_ref::<Vec<String>>()
                .unwrap(),
            &vec!["0", "1", "2", "3", "4"]
        );
    }

    #[test]
    fn test_for_loop_with_tools() {
        let code = textwrap::dedent(
            r#"
        for i in range(5):
            search = duckduckgo_search(query=i)
            print(search)
        "#,
        );
        let mut state = HashMap::new();
        let tools: Vec<Box<dyn AnyTool>> = vec![Box::new(DuckDuckGoSearchTool::new())];
        let _ = evaluate_python_code(&code, tools, &mut state).unwrap();
    }

    #[test]
    fn test_evaluate_python_code_with_dict() {
        let code = textwrap::dedent(
            r#"
        my_dict = {'a': "1", 'b': "2", 'c': "3"}
        print(f"my_dict['a'] is {my_dict['a']}")
        "#,
        );
        let mut state = HashMap::new();
        let result = evaluate_python_code(&code, vec![], &mut state).unwrap();
        assert_eq!(result, "my_dict['a'] is 1");

        let code = textwrap::dedent(
            r#"
dinner_places = [
    {
        "title": "25 Best Restaurants in Berlin, By Local Foodies",
        "url": "https://www.timeout.com/berlin/restaurants/best-restaurants-in-berlin"
    },
    {
        "title": "The 38 Best Berlin Restaurants - Eater",
        "url": "https://www.eater.com/maps/best-restaurants-berlin"
    },
    {
        "title": "THE 10 BEST Restaurants in Berlin - Tripadvisor",
        "url": "https://www.tripadvisor.com/Restaurants-g187323-Berlin.html"
    },
    {
        "title": "12 Unique Restaurants in Berlin",
        "url": "https://www.myglobalviewpoint.com/unique-restaurants-in-berlin/"
    },
    {
        "title": "Berlin's best restaurants: 101 places to eat right now",
        "url": "https://www.the-berliner.com/food/best-restaurants-berlin-101-places-to-eat/"
    }
]

for place in dinner_places:
    print(f"{place['title']}: {place['url']}")
        "#,
        );
        let mut local_python_interpreter = LocalPythonInterpreter::new(vec![]);
        let (_, execution_logs) = local_python_interpreter
            .forward(&code)
            .unwrap();
        assert_eq!(execution_logs, "25 Best Restaurants in Berlin, By Local Foodies: https://www.timeout.com/berlin/restaurants/best-restaurants-in-berlin\nThe 38 Best Berlin Restaurants - Eater: https://www.eater.com/maps/best-restaurants-berlin\nTHE 10 BEST Restaurants in Berlin - Tripadvisor: https://www.tripadvisor.com/Restaurants-g187323-Berlin.html\n12 Unique Restaurants in Berlin: https://www.myglobalviewpoint.com/unique-restaurants-in-berlin/\nBerlin's best restaurants: 101 places to eat right now: https://www.the-berliner.com/food/best-restaurants-berlin-101-places-to-eat/");

        let code = textwrap::dedent(
            r#"
movies = [
    {"title": "Babygirl", "showtimes": ["12:50 pm", "6:20 pm"]},
    {"title": "Better Man", "showtimes": ["9:20 pm"]},
    {"title": "La acompañante", "showtimes": ["3:40 pm", "6:30 pm", "9:10 pm"]},
    {"title": "Amenaza en el aire", "showtimes": ["9:30 pm"]},
    {"title": "Juf Braaksel en de Geniale Ontsnapping", "showtimes": ["12:30 pm"]},
    {"title": "Juffrouw Pots", "showtimes": ["10:35 am", "3:50 pm"]},
    {"title": "K3 en Het Lied van de Zeemeermin", "showtimes": ["10:00 am"]},
    {"title": "Marked Men", "showtimes": ["2:50 pm", "6:50 pm"]},
    {"title": "Vaiana 2", "showtimes": ["11:10 am", "12:40 pm"]},
    {"title": "Mufasa: El rey león", "showtimes": ["10:20 am", "3:10 pm", "9:00 pm"]},
    {"title": "Paddington: Aventura en la selva", "showtimes": ["12:20 pm", "3:30 pm", "6:10 pm"]},
    {"title": "Royal Opera House: The Tales of Hoffmann", "showtimes": ["1:30 pm"]},
    {"title": "The Growcodile", "showtimes": ["10:10 am"]},
    {"title": "Vivir el momento", "showtimes": ["5:20 pm"]},
    {"title": "Wicked", "showtimes": ["7:00 pm"]},
    {"title": "Woezel & Pip op Avontuur in de Tovertuin", "showtimes": ["10:30 am", "1:50 pm"]}
]

for movie in movies:
    print(f"{movie['title']}: {', '.join(movie['showtimes'])}")

        "#,
        );
        let mut local_python_interpreter = LocalPythonInterpreter::new(vec![]);
        let (_, _) = local_python_interpreter
            .forward(&code)
            .unwrap();

        let code = textwrap::dedent(
            r#"
urls = [
    "https://www.tripadvisor.com/Restaurants-g187323-Berlin.html",
    "https://www.timeout.com/berlin/restaurants/best-restaurants-in-berlin"
]

for url in urls:
    page_content = duckduckgo_search(url)
    print(page_content)
    print("\n" + "="*80 + "\n")  # Print separator between pages        
    "#,
        );
        let mut state = HashMap::new();
        let tools: Vec<Box<dyn AnyTool>> = vec![Box::new(DuckDuckGoSearchTool::new())];
        let _ = evaluate_python_code(&code, tools, &mut state).unwrap();
    }

    #[test]
    fn test_evaluate_python_code_with_list_comprehension() {
        let code = textwrap::dedent(
            r#"
        a = [1,2,3]
        print([x for x in a])
    "#,
        );
        let mut state = HashMap::new();
        let _ = evaluate_python_code(&code, vec![], &mut state).unwrap();
        assert_eq!(
            state
                .get("print_logs")
                .unwrap()
                .downcast_ref::<Vec<String>>()
                .unwrap(),
            &vec!["[1, 2, 3]"]
        );
    }

    #[test]
    fn test_evaluate_python_code_append_to_list() {
        let code = textwrap::dedent(
            r#"
        a = [1,2,3]
        a.append(4)
        print(a)
    "#,
        );
        let mut state = HashMap::new();
        let _ = evaluate_python_code(&code, vec![], &mut state).unwrap();
        assert_eq!(
            state
                .get("print_logs")
                .unwrap()
                .downcast_ref::<Vec<String>>()
                .unwrap(),
            &vec!["[1, 2, 3, 4]"]
        );

        let code = textwrap::dedent(
            r#"
urls = [
    "https://www.imdb.com/showtimes/cinema/ES/ci1028808/ES/08520",
    "https://en.pathe.nl/bioscoopagenda",
    "https://www.filmvandaag.nl/bioscoop?filter=64"
]
movies = []
for url in urls:
    page_content = url
    movies.append(page_content)

print(movies)
    "#,
        );
        let mut state = HashMap::new();
        let tools: Vec<Box<dyn AnyTool>> = vec![Box::new(VisitWebsiteTool::new())];
        let _ = evaluate_python_code(&code, tools, &mut state).unwrap();
        assert_eq!(
            state
                .get("print_logs")
                .unwrap()
                .downcast_ref::<Vec<String>>()
                .unwrap(),
            &vec!["[https://www.imdb.com/showtimes/cinema/ES/ci1028808/ES/08520, https://en.pathe.nl/bioscoopagenda, https://www.filmvandaag.nl/bioscoop?filter=64]"]
        );
    }

    #[test]
    fn test_evaluate_python_code_with_error() {
        let code = textwrap::dedent(
            r#"
guidelines = (
    "To avoid being blocked by websites, use the following guidelines for user agent strings:\n"
    "1. Use a valid browser user agent to mimic a real web browser.\n"
    "2. Rotate User-Agent headers for each outgoing request to prevent identification as a bot.\n"
    "3. Avoid using generic user-agent strings like 'Python Requests Library' or an empty UA string.\n"
    "4. Use a user agent string that includes information about the browser, operating system, and other parameters.\n"
    "5. Understand that websites use user agent strings to organize protection against malicious actions, including parsing blocks."
)

    "#,
        );
        let code_2 = textwrap::dedent(
            r#"
            print(guidelines)
            "#,
        );
        let tools: Vec<Box<dyn AnyTool>> = vec![Box::new(VisitWebsiteTool::new())];
        let mut local_python_interpreter = LocalPythonInterpreter::new(tools);
        let (_, logs) = local_python_interpreter
            .forward(&code)
            .unwrap();
        println!("logs: {:?}", logs);
        let (_, logs_2) = local_python_interpreter
            .forward(&code_2)
            .unwrap();
        println!("logs_2: {:?}", logs_2);
    }
}
