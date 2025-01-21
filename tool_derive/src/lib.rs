use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, DeriveInput};

#[proc_macro_derive(BaseTool)]
pub fn base_tool_derive(input: TokenStream) -> TokenStream {
    let ast = parse_macro_input!(input as DeriveInput);
    let name = &ast.ident;
    let gen = quote! {
        impl Tool for #name {
            fn name(&self) -> &'static str { self.name }
            fn description(&self) -> &'static str { self.description }
            fn inputs(&self) -> &HashMap<&'static str, HashMap<&'static str, String>> { &self.inputs }
            fn output_type(&self) -> &'static str { self.output_type }
            fn is_initialized(&self) -> bool { self.is_initialized }
            fn box_clone(&self) -> Box<dyn Tool> { Box::new(self.clone()) }
            fn forward(&self, _arguments: HashMap<String, String>) -> Result<String> {
                unimplemented!("Base tool does not implement forward")
            }
        }
    };
    gen.into()
}

#[proc_macro_derive(ToolDerive)]
pub fn tool_derive(input: TokenStream) -> TokenStream {
    let ast = parse_macro_input!(input as syn::DeriveInput);
    let name = &ast.ident;
    let gen = quote! {
        impl Tool for #name {
            fn name(&self) -> &'static str { self.tool.name }
            fn description(&self) -> &'static str { self.tool.description }
            fn inputs(&self) -> &HashMap<&'static str, HashMap<&'static str, String>> { &self.tool.inputs }
            fn output_type(&self) -> &'static str { self.tool.output_type }
            fn is_initialized(&self) -> bool { self.tool.is_initialized }
            fn box_clone(&self) -> Box<dyn Tool> { Box::new(self.clone()) }
            fn forward(&self, arguments: HashMap<String, String>) -> Result<String> {
                self.forward(arguments)
            }
        }
    };
    gen.into()
}
