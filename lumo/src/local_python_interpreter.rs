use crate::errors::InterpreterError;
use crate::tools::tool_traits::AsyncTool;
use crate::tools::ToolInfo;
use anyhow::Result;
use pyo3::types::{IntoPyDict, PyDict, PyModule, PyTuple};
use pyo3::{prelude::*, IntoPyObjectExt};
use rustpython_parser::ast::{
    bigint::{BigInt, Sign},
    Constant,
};
use serde_json::{self, json, Value};
use std::collections::HashMap;
use std::ffi::CString;
use tokio::runtime::Runtime;

impl From<PyErr> for InterpreterError {
    fn from(err: PyErr) -> Self {
        Python::with_gil(|py| {
            if err.is_instance_of::<pyo3::exceptions::PyStopIteration>(py) {
                let value = err
                    .value(py)
                    .extract::<String>()
                    .unwrap_or_else(|_| err.value(py).to_string());
                InterpreterError::FinalAnswer(value)
            } else {
                InterpreterError::RuntimeError(err.to_string())
            }
        })
    }
}

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

#[derive(Debug)]
pub enum CustomConstant {
    Int(BigInt),
    Float(f64),
    Str(String),
    Bool(bool),
    Tuple(Vec<CustomConstant>),
    PyObj(Py<PyAny>),
    Dict(Vec<String>, Vec<CustomConstant>),
}

impl Clone for CustomConstant {
    fn clone(&self) -> Self {
        match self {
            Self::Int(i) => Self::Int(i.clone()),
            Self::Float(f) => Self::Float(*f),
            Self::Str(s) => Self::Str(s.clone()),
            Self::Bool(b) => Self::Bool(*b),
            Self::Tuple(t) => Self::Tuple(t.clone()),
            Self::PyObj(p) => Python::with_gil(|py| Self::PyObj(p.clone_ref(py))),
            Self::Dict(k, v) => Self::Dict(k.clone(), v.clone()),
        }
    }
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

impl<'py> IntoPyObject<'py> for CustomConstant {
    type Target = PyAny;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        match self {
            CustomConstant::Int(i) => convert_bigint_to_i64(&i).into_bound_py_any(py),
            CustomConstant::Float(f) => f.into_bound_py_any(py),
            CustomConstant::Str(s) => s.into_bound_py_any(py),
            CustomConstant::Bool(b) => b.into_bound_py_any(py),
            CustomConstant::Tuple(t) => {
                let py_list = t
                    .iter()
                    .map(|x| x.clone().into_bound_py_any(py).unwrap())
                    .collect::<Vec<Bound<'py, PyAny>>>();
                PyTuple::new(py, py_list)?.into_bound_py_any(py)
            }
            CustomConstant::PyObj(obj) => obj.into_bound_py_any(py),
            CustomConstant::Dict(keys, values) => {
                let dict = PyDict::new(py);
                for (key, value) in keys.iter().zip(values.iter()) {
                    dict.set_item(key, value.clone().into_bound_py_any(py)?)
                        .unwrap_or_default();
                }
                dict.into_bound_py_any(py)
            }
        }
    }
}

#[pyclass]
pub struct PythonToolFunction {
    tool: Box<dyn Fn(Value) -> Result<CustomConstant, InterpreterError> + Send + Sync>,
    tool_info: ToolInfo,
}

#[pymethods]
impl PythonToolFunction {
    #[pyo3(signature = (*args, **kwargs))]
    pub fn __call__(
        &self,
        _py: Python<'_>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<CustomConstant> {
        let mut arguments = HashMap::new();

        // Handle args based on whether it's a single argument or multiple arguments
        if let Ok(tuple) = args.downcast::<PyTuple>() {
            // Multiple arguments case
            Python::with_gil(|py| {
                for (i, arg) in tuple.iter().enumerate() {
                    if i < self.tool_info.get_parameter_names().len() {
                        let value = extract_constant_from_pyobject(&arg, py).unwrap().str();
                        arguments.insert(self.tool_info.get_parameter_names()[i].clone(), value);
                    }
                }
                Ok::<_, PyErr>(())
            })?;
        } else {
            // Single argument case
            if !self.tool_info.get_parameter_names().is_empty() {
                Python::with_gil(|py| {
                    let value = extract_constant_from_pyobject(args, py).unwrap().str();
                    arguments.insert(self.tool_info.get_parameter_names()[0].clone(), value);
                    Ok::<_, PyErr>(())
                })?;
            }
        }

        // Handle keyword arguments
        if let Some(kwargs) = kwargs {
            Python::with_gil(|py| {
                for (key, value) in kwargs.iter() {
                    let key = key.extract::<String>()?;
                    let value = extract_constant_from_pyobject(&value, py).unwrap().str();
                    arguments.insert(key, value);
                }
                Ok::<_, PyErr>(())
            })?;
        }

        // Convert arguments to JSON Value
        let args_json = json!(arguments);
        // Call the tool function
        (self.tool)(args_json).map_err(|e| match e {
            InterpreterError::FinalAnswer(answer) => {
                PyErr::new::<pyo3::exceptions::PyStopIteration, _>(answer)
            }
            _ => PyErr::new::<pyo3::exceptions::PyException, _>(e.to_string()),
        })
    }
}

fn setup_custom_tools(
    tools: &[Box<dyn AsyncTool>],
    runtime: &Runtime,
) -> HashMap<String, PythonToolFunction> {
    let mut tools_map = HashMap::new();
    for tool in tools {
        let tool = tool.clone_box();
        let tool_name = tool.name().to_string();
        let tool_info = tool.tool_info();
        let runtime = runtime.handle().clone();
        tools_map.insert(
            tool_name.clone(),
            PythonToolFunction {
                tool: Box::new(move |args: Value| {
                    if tool_name == "final_answer" {
                        return Err(InterpreterError::FinalAnswer(
                            args.get("answer").unwrap().as_str().unwrap().to_string(),
                        ));
                    }

                    let tool_clone = tool.clone_box();
                    // Execute the async operation synchronously
                    let result = runtime.block_on(async { tool_clone.forward_json(args).await });

                    match result {
                        Ok(result) => Ok(CustomConstant::Str(result)),
                        Err(e) => Err(InterpreterError::RuntimeError(e.to_string())),
                    }
                }),
                tool_info,
            },
        );
    }
    tools_map
}

fn convert_bigint_to_i64(i: &BigInt) -> i64 {
    let i = i.to_u32_digits();
    let num = i.1.iter().fold(0i64, |acc, &d| acc * (1 << 32) + d as i64);
    match i.0 {
        Sign::Minus => -num,
        Sign::NoSign | Sign::Plus => num,
    }
}

fn extract_constant_from_pyobject(
    obj: &Bound<'_, PyAny>,
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
    } else if let Ok(dict_value) = obj.downcast::<PyDict>() {
        let keys = dict_value
            .iter()
            .map(|(key, _)| key.extract::<String>())
            .collect::<Result<Vec<String>, _>>()?;
        let values = dict_value
            .iter()
            .map(|(_, value)| extract_constant_from_pyobject(&value, py))
            .collect::<Result<Vec<CustomConstant>, _>>()?;
        Ok(CustomConstant::Dict(keys, values))
    } else {
        Ok(CustomConstant::PyObj(obj.into_bound_py_any(py)?.into()))
    }
}

fn evaluate_python_code(
    code: &str,
    custom_tools: Option<&[Box<dyn AsyncTool>]>,
    static_tools: &HashMap<&'static str, &'static str>,
    state: &mut HashMap<String, Py<PyAny>>,
    runtime: Option<&Runtime>,
) -> Result<String, InterpreterError> {
    let custom_tools = match custom_tools {
        Some(tools) => Some(setup_custom_tools(tools, runtime.unwrap())),
        None => None,
    };
    let code = code.to_string();
    let static_tools = static_tools.clone();
    let state_clone: HashMap<String, Py<PyAny>> = Python::with_gil(|py| {
        state
            .iter()
            .map(|(k, v)| (k.clone(), v.clone_ref(py)))
            .collect()
    });

    // Move Python operations to a separate thread using std::thread
    let handle = std::thread::spawn(move || {
        Python::with_gil(|py| -> PyResult<(String, HashMap<String, PyObject>)> {
            let locals = state_clone.into_py_dict(py)?;
            let globals = PyDict::new(py);

            // Add base Python tools to globals
            for name in static_tools.keys() {
                if let Ok(builtin) = {
                    let cmd = CString::new(format!("__builtins__.{}", name)).unwrap();
                    py.eval(&cmd, None, None)
                } {
                    globals.set_item(name, builtin)?;
                }
            }

            // Add custom tools to globals
            if let Some(tools) = custom_tools {
                for (name, tool) in tools {
                    globals.set_item(name.to_string(), tool.into_bound_py_any(py)?)?;
                }
            }

            // Add math module functions that are in base_tools
            let math = PyModule::import(py, "math")?;
            globals.set_item("math", math)?;

            // Setup StringIO for capturing output
            let io = PyModule::import(py, "io")?;
            let string_io = io.call_method0("StringIO")?;
            globals.set_item("stdout", string_io.clone())?;

            // Redirect stdout
            let cmd = CString::new(format!("import sys; sys.stdout = stdout")).unwrap();
            py.run(&cmd, Some(&globals), None)?;

            let code_str = CString::new(code).unwrap();
            // Run the user code with restricted globals
            py.run(&code_str, Some(&globals), Some(&locals))?;

            // Get the output
            locals.set_item(
                "print_logs",
                string_io.call_method0("getvalue")?.extract::<String>()?,
            )?;

            // Create new state from locals
            let mut new_state = HashMap::new();
            for key in locals.keys() {
                let value = locals.get_item(key.clone()).unwrap();
                new_state.insert(key.to_string(), value.into_pyobject(py)?.into());
            }

            let output = locals
                .get_item("print_logs")
                .and_then(|obj| Ok(obj.unwrap().extract::<String>().unwrap_or_default()))
                .unwrap_or_default();

            Ok((output, new_state))
        })
    });

    // Convert the JoinHandle result into our Result type
    match handle.join() {
        Ok(result) => {
            let (output, new_state) = result?;
            // Update the original state with new values
            state.clear();
            state.extend(new_state);
            Ok(output)
        }
        Err(e) => Err(InterpreterError::RuntimeError(format!(
            "Thread panicked: {:?}",
            e
        ))),
    }
}

#[derive(Debug)]
pub struct LocalPythonInterpreter {
    static_tools: HashMap<&'static str, &'static str>,
    custom_tools: Option<Vec<Box<dyn AsyncTool>>>,
    state: HashMap<String, PyObject>,
    runtime: Option<Runtime>,
}

impl LocalPythonInterpreter {
    pub fn new(
        custom_tools: Option<&[Box<dyn AsyncTool>]>,
        static_tools: Option<HashMap<&'static str, &'static str>>,
    ) -> Self {
        let static_tools = static_tools.unwrap_or_else(get_base_python_tools);
        let (runtime, custom_tools) = if let Some(tools) = custom_tools {
            (
                Some(Runtime::new().unwrap()),
                Some(tools.iter().map(|tool| tool.clone_box()).collect()),
            )
        } else {
            (None, None)
        };

        Self {
            static_tools,
            custom_tools,
            state: HashMap::new(),
            runtime,
        }
    }

    pub fn forward(&mut self, code: &str) -> Result<(String, String), InterpreterError> {
        let execution_logs = evaluate_python_code(
            code,
            self.custom_tools.as_deref(),
            &self.static_tools,
            &mut self.state,
            self.runtime.as_ref(),
        )?;

        Ok(("".to_string(), execution_logs.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::{DuckDuckGoSearchTool, FinalAnswerTool, VisitWebsiteTool};

    #[test]
    fn test_evaluate_python_code() {
        let code = "print('Hello, world!')";
        let mut interpreter = LocalPythonInterpreter::new(None, None);
        let (_, execution_logs) = interpreter.forward(&code).unwrap();
        assert_eq!(execution_logs, "Hello, world!\n");
    }

    #[test]
    fn test_evaluate_python_code_with_joined_str() {
        let code = r#"word = 'strawberry'
r_count = word.count('r')
print(f"The letter 'r' appears {r_count} times in the word '{word}'.")"#;
        let mut interpreter = LocalPythonInterpreter::new(None, None);
        let (_, execution_logs) = interpreter.forward(&code).unwrap();
        assert_eq!(
            execution_logs,
            "The letter 'r' appears 3 times in the word 'strawberry'.\n"
        );
    }

    #[test]
    fn test_final_answer_execution() {
        let tools: Vec<Box<dyn AsyncTool>> = vec![Box::new(FinalAnswerTool::new())];
        let mut interpreter = LocalPythonInterpreter::new(Some(&tools), None);
        let result = interpreter.forward("final_answer('Hello, world!')");
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
        let mut interpreter = LocalPythonInterpreter::new(None, None);
        let (_, execution_logs) = interpreter.forward(&code).unwrap();
        assert_eq!(execution_logs, "a\n");

        let code = textwrap::dedent(
            r#"
        word = 'strawberry'
        print(word[-3])"#,
        );
        let mut interpreter = LocalPythonInterpreter::new(None, None);
        let (_, execution_logs) = interpreter.forward(&code).unwrap();
        assert_eq!(execution_logs, "r\n");

        let code = textwrap::dedent(
            r#"
        word = 'strawberry'
        print(word[9])"#,
        );
        let mut interpreter = LocalPythonInterpreter::new(None, None);
        let (_, execution_logs) = interpreter.forward(&code).unwrap();
        assert_eq!(execution_logs, "y\n");

        let code = textwrap::dedent(
            r#"
        word = 'strawberry'
        print(word[10])"#,
        );
        let mut interpreter = LocalPythonInterpreter::new(None, None);
        let result = interpreter.forward(&code);
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
        let mut interpreter = LocalPythonInterpreter::new(None, None);
        let (_, execution_logs) = interpreter.forward(&code).unwrap();
        assert_eq!(execution_logs, "2\n");

        let code = textwrap::dedent(
            r#"
        numbers = [1, 2, 3, 4, 5]
        print(numbers[-5])"#,
        );
        let mut interpreter = LocalPythonInterpreter::new(None, None);
        let (_, execution_logs) = interpreter.forward(&code).unwrap();
        assert_eq!(execution_logs, "1\n");

        let code = textwrap::dedent(
            r#"
        numbers = [1, 2, 3, 4, 5]
        print(numbers[-6])"#,
        );
        let mut interpreter = LocalPythonInterpreter::new(None, None);
        let result = interpreter.forward(&code);
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
        let mut interpreter = LocalPythonInterpreter::new(None, None);
        let (_, execution_logs) = interpreter.forward(&code).unwrap();
        assert_eq!(execution_logs, "[2, 3]\n");

        let code = textwrap::dedent(
            r#"
        numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        print(numbers[1:5:2])"#,
        );
        let mut interpreter = LocalPythonInterpreter::new(None, None);
        let (_, execution_logs) = interpreter.forward(&code).unwrap();
        assert_eq!(execution_logs, "[2, 4]\n");

        let code = textwrap::dedent(
            r#"
        numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        print(numbers[5:1:-2])"#,
        );
        let mut interpreter = LocalPythonInterpreter::new(None, None);
        let (_, execution_logs) = interpreter.forward(&code).unwrap();
        assert_eq!(execution_logs, "[6, 4]\n");

        let code = textwrap::dedent(
            r#"
        word = 'strawberry'
        print(word[::-1])"#,
        );
        let mut interpreter = LocalPythonInterpreter::new(None, None);
        let (_, execution_logs) = interpreter.forward(&code).unwrap();
        assert_eq!(execution_logs, "yrrebwarts\n");

        let code = textwrap::dedent(
            r#"
        numbers = [1, 2, 3, 4, 5]
        print(numbers[::-1])"#,
        );
        let mut interpreter = LocalPythonInterpreter::new(None, None);
        let (_, execution_logs) = interpreter.forward(&code).unwrap();
        assert_eq!(execution_logs, "[5, 4, 3, 2, 1]\n");
    }

    #[test]
    fn test_for_loop() {
        let code = textwrap::dedent(
            r#"
        for i in range(5):
            print(i)
        "#,
        );
        let mut interpreter = LocalPythonInterpreter::new(None, None);
        let (_, execution_logs) = interpreter.forward(&code).unwrap();
        assert_eq!(execution_logs, "0\n1\n2\n3\n4\n");
    }

    #[test]
    fn test_for_loop_with_tools() {
        let code = textwrap::dedent(
            r#"
        for i in range(3):
            search = duckduckgo_search(query=i)
            print(search)
        "#,
        );
        let tools: Vec<Box<dyn AsyncTool>> = vec![Box::new(DuckDuckGoSearchTool::new())];
        let mut interpreter = LocalPythonInterpreter::new(Some(&tools), None);
        let (_, _) = interpreter.forward(&code).unwrap();
    }

    #[test]
    fn test_evaluate_python_code_with_dict() {
        let code = textwrap::dedent(
            r#"
        my_dict = {'a': "1", 'b': "2", 'c': "3"}
        print(f"my_dict['a'] is {my_dict['a']}")
        "#,
        );
        let mut interpreter = LocalPythonInterpreter::new(None, None);
        let (_, execution_logs) = interpreter.forward(&code).unwrap();
        assert_eq!(execution_logs, "my_dict['a'] is 1\n");

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
        let mut local_python_interpreter = LocalPythonInterpreter::new(None, None);
        let (_, execution_logs) = local_python_interpreter.forward(&code).unwrap();
        assert_eq!(execution_logs, "25 Best Restaurants in Berlin, By Local Foodies: https://www.timeout.com/berlin/restaurants/best-restaurants-in-berlin\nThe 38 Best Berlin Restaurants - Eater: https://www.eater.com/maps/best-restaurants-berlin\nTHE 10 BEST Restaurants in Berlin - Tripadvisor: https://www.tripadvisor.com/Restaurants-g187323-Berlin.html\n12 Unique Restaurants in Berlin: https://www.myglobalviewpoint.com/unique-restaurants-in-berlin/\nBerlin's best restaurants: 101 places to eat right now: https://www.the-berliner.com/food/best-restaurants-berlin-101-places-to-eat/\n");

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
        let tools: Vec<Box<dyn AsyncTool>> = vec![Box::new(DuckDuckGoSearchTool::new())];
        let mut interpreter = LocalPythonInterpreter::new(Some(&tools), None);
        let (_, _) = interpreter.forward(&code).unwrap();
    }

    #[test]
    fn test_evaluate_python_code_with_list_comprehension() {
        let code = textwrap::dedent(
            r#"
        a = [1,2,3]
        print([x for x in a])
    "#,
        );
        let mut interpreter = LocalPythonInterpreter::new(None, None);
        let (_, execution_logs) = interpreter.forward(&code).unwrap();
        assert_eq!(execution_logs, "[1, 2, 3]\n");
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
        let mut python_interpreter = LocalPythonInterpreter::new(None, None);
        let (_, execution_logs) = python_interpreter.forward(&code).unwrap();
        assert_eq!(execution_logs, "[1, 2, 3, 4]\n");

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
        let tools: Vec<Box<dyn AsyncTool>> = vec![Box::new(VisitWebsiteTool::new())];
        let mut interpreter = LocalPythonInterpreter::new(Some(&tools), None);
        let (_, execution_logs) = interpreter.forward(&code).unwrap();
        assert_eq!(
            execution_logs,
            "['https://www.imdb.com/showtimes/cinema/ES/ci1028808/ES/08520', 'https://en.pathe.nl/bioscoopagenda', 'https://www.filmvandaag.nl/bioscoop?filter=64']\n"
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
        let tools: Vec<Box<dyn AsyncTool>> = vec![Box::new(VisitWebsiteTool::new())];
        let mut local_python_interpreter = LocalPythonInterpreter::new(Some(&tools), None);
        let (_, logs) = local_python_interpreter.forward(&code).unwrap();
        println!("logs: {:?}", logs);
        let (_, logs_2) = local_python_interpreter.forward(&code_2).unwrap();
        println!("logs_2: {:?}", logs_2);
    }
}
