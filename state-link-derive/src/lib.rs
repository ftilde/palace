use proc_macro::{self, TokenStream};
use quote::quote;
use syn::{parse_macro_input, parse_quote, Data, DeriveInput, Fields, GenericParam, Generics};

#[proc_macro_derive(State)]
pub fn derive_state(input: TokenStream) -> TokenStream {
    let gen_python = cfg!(feature = "python");
    derive_wrapper(input, gen_python)
}

#[proc_macro_derive(StateNoPy)]
pub fn derive_state_no_py(input: TokenStream) -> TokenStream {
    derive_wrapper(input, false)
}

fn derive_wrapper(input: TokenStream, gen_python: bool) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);

    // Used in the quasi-quotation below as `#name`.
    let name = input.ident;

    let visibility = input.vis;

    // Add a bound `T: HeapSize` to every type parameter T.
    let generics = add_trait_bounds(input.generics.clone());
    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();

    let data = input.data;

    let (store, load, write, handle_access) = match &data {
        Data::Struct(s) => match &s.fields {
            Fields::Named(fields) => {
                let field_load = fields.named.iter().map(|f| {
                    let name = &f.ident;
                    let ty = &f.ty;
                    quote! {
                        #name: {
                            let field_name = stringify!(#name);
                            let loc = map.get(field_name).ok_or(state_link::Error::MissingField(field_name))?;
                            <#ty>::load(store, *loc)?
                        },
                    }
                });

                let load_code = quote! {
                    if let state_link::ResolveResult::Struct(map) = store.to_val(location)? {
                        Ok(#name {
                            #(#field_load)*
                        })
                    } else {
                        Err(state_link::Error::IncorrectType)
                    }
                };

                let field_store = fields.named.iter().map(|f| {
                    let name = &f.ident;
                    quote! {
                        map.insert(stringify!(#name).to_owned(), ::state_link::State::store(&self.#name, store));
                    }
                });

                let store_code = quote! {
                    let mut map = ::state_link::Map::default();
                    #(#field_store)*

                    store.push(::state_link::Node::Dir(map))
                };

                let field_write = fields.named.iter().map(|f| {
                    let name = &f.ident;
                    quote! {
                        {
                            let field_name = stringify!(#name);
                            let loc = map.get(field_name).ok_or(state_link::Error::MissingField(field_name))?;
                            ::state_link::State::write(&self.#name, store, *loc)?;
                        }
                    }
                });

                let write_code = quote! {
                    let map = if let ::state_link::ResolveResult::Struct(map) = store.to_val(at)? {
                        map.clone()
                    } else {
                        return Err(::state_link::Error::IncorrectType);
                    };

                    #(#field_write)*;

                    Ok(())
                };

                let handle_accesses = fields.named.iter().map(|f| {
                    let name = &f.ident;
                    let ty = &f.ty;
                    quote! {
                        #visibility fn #name(&self) -> <#ty as ::state_link::State>::NodeHandle {
                            <<#ty as ::state_link::State>::NodeHandle as state_link::NodeHandle>::pack(self.inner.named(stringify!(#name).to_owned()))
                        }
                    }
                });

                let handle_access = quote! { #(#handle_accesses)* };

                (store_code, load_code, write_code, handle_access)
            }
            Fields::Unnamed(fields) => {
                let field_load = fields.unnamed.iter().enumerate().map(|(num, f)| {
                    let ty = &f.ty;
                    quote! {
                        {
                            let loc = seq.get(#num).ok_or(state_link::Error::SeqTooShort)?;
                            <#ty>::load(store, *loc)?
                        },
                    }
                });

                let load_code = quote! {
                    if let state_link::ResolveResult::Seq(seq) = store.to_val(location)? {
                        Ok(#name (
                            #(#field_load)*
                        ))
                    } else {
                        Err(state_link::Error::IncorrectType)
                    }
                };

                let field_store = fields.unnamed.iter().enumerate().map(|(num, _f)| {
                    let num = syn::Index::from(num);
                    quote! {
                        seq.push(::state_link::State::store(&self.#num, store));
                    }
                });

                let store_code = quote! {
                    let mut seq = Vec::default();
                    #(#field_store)*

                    store.push(::state_link::Node::Seq(seq))
                };

                let field_write = fields.unnamed.iter().enumerate().map(|(num, _f)| {
                    let num = syn::Index::from(num);
                    quote! {
                        {
                            let loc = seq.get(#num).ok_or(state_link::Error::SeqTooShort)?;
                            ::state_link::State::write(&self.#num, store, *loc)?;
                        }
                    }
                });

                let num_fields = fields.unnamed.len();

                let write_code = quote! {
                    let mut seq = if let ::state_link::ResolveResult::Seq(seq) = store.to_val(at)? {
                        seq.clone()
                    } else {
                        return Err(::state_link::Error::IncorrectType);
                    };

                    if seq.len() != #num_fields {
                        return Err(::state_link::Error::IncorrectType);
                    }

                    #(#field_write)*;

                    store.write_at(::state_link::Node::Seq(seq), at)
                };

                let handle_accesses = fields.unnamed.iter().enumerate().map(|(num, f)| {
                    let ty = &f.ty;

                    let fn_name = quote::format_ident!("elm{}", num);

                    quote! {
                        pub fn #fn_name(&self) -> <#ty as ::state_link::State>::NodeHandle {
                            <<#ty as ::state_link::State>::NodeHandle as state_link::NodeHandle>::pack(self.inner.index(#num))
                        }
                    }
                });

                let handle_access = quote! { #(#handle_accesses)* };

                (store_code, load_code, write_code, handle_access)
            }
            Fields::Unit => {
                let load_code = quote! {
                    if let state_link::ResolveResult::Atom(state_link::Value::Unit) = store.to_val(location)? {
                        Ok(#name)
                    } else {
                        Err(state_link::Error::IncorrectType)
                    }
                };

                let store_code = quote! {
                    store.push(::state_link::Node::Val(::state_link::Value::Unit))
                };

                let write_code = quote! {
                    Ok(())
                };

                let handle_access = quote! {};

                (store_code, load_code, write_code, handle_access)
            }
        },
        Data::Enum(e) => {
            let str_to_variant = e.variants.iter().map(|v| {
                assert_eq!(
                    v.fields.len(),
                    0,
                    "Only simple enums without data are supported"
                );
                let name = &v.ident;
                quote! {
                    if s == stringify!(#name) {
                        Ok(Self::#name)
                    } else
                }
            });
            let load_code = quote! {
                if let state_link::ResolveResult::Atom(state_link::Value::String(s)) = store.to_val(location)? {
                    #(#str_to_variant)*
                    {
                        Err(state_link::Error::UnknownVariant)
                    }
                } else {
                    Err(state_link::Error::IncorrectType)
                }
            };

            let variant_to_str = e.variants.iter().map(|v| {
                let name = &v.ident;
                quote! {
                    Self::#name => stringify!(#name),
                }
            });
            let variant_to_str2 = variant_to_str.clone();

            let store_code = quote! {
                let s = match self {
                    #(#variant_to_str)*
                };
                store.push(::state_link::Node::Val(::state_link::Value::String(s.to_owned())))
            };

            let write_code = quote! {
                let s = match self {
                    #(#variant_to_str2)*
                };

                store.write_at(::state_link::Node::Val(::state_link::Value::String(s.to_owned())), at)
            };

            let handle_access = quote! {};

            (store_code, load_code, write_code, handle_access)
        }
        Data::Union(_) => unimplemented!(),
    };

    let node_handle_name = quote::format_ident!("__NodeHandle_{}", name);

    let has_generics = !input.generics.params.is_empty();

    let core_output = quote! {
        #[allow(non_camel_case_types)]
        #visibility struct #node_handle_name #impl_generics {
            inner: ::state_link::GenericNodeHandle,
            _marker: ::std::marker::PhantomData<#name #ty_generics>
        }

        impl #impl_generics ::state_link::NodeHandle for #node_handle_name #ty_generics {
            type NodeType = #name #ty_generics;
            fn pack(inner: ::state_link::GenericNodeHandle) -> Self {
                Self {
                    inner,
                    _marker: ::std::marker::PhantomData,
                }
            }
            fn unpack(&self) -> &::state_link::GenericNodeHandle {
                &self.inner
            }
        }

        impl #impl_generics #node_handle_name #ty_generics #where_clause {
            #handle_access
        }

        impl #impl_generics ::state_link::State for #name #ty_generics #where_clause {
            type NodeHandle = #node_handle_name #ty_generics;
            fn store(&self, store: &mut ::state_link::Store) -> ::state_link::NodeRef {
                #store
            }

            fn load(store: &::state_link::Store, location: ::state_link::NodeRef) -> ::state_link::Result<Self> {
                #load
            }

            fn write(&self, store: &mut ::state_link::Store, at: ::state_link::NodeRef) -> ::state_link::Result<()> {
                #write
            }
        }
    };

    let generics = add_trait_bounds_py(input.generics);
    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();

    let inner_node_handle_name = node_handle_name;
    let node_handle_name = quote::format_ident!("__PyNodeHandle_{}", name);

    let python_code = if gen_python && !has_generics {
        match data {
            Data::Struct(s) => {
                let handle_access = match s.fields {
                    Fields::Named(fields) => {
                        let handle_accesses = fields.named.iter().map(|f| {
                        let name = &f.ident;
                        let name = name.as_ref().unwrap();
                        let ty = &f.ty;
                        let func_name = quote::format_ident!("{}_py", name);
                        let inside_py_name_str = format!("{}", name);
                        quote! {
                            #[pyo3(name = #inside_py_name_str)]
                            fn #func_name(&self, py: pyo3::Python) -> pyo3::PyObject {
                                <#ty as state_link::py::PyState>::build_handle(py, self.inner.inner.named(stringify!(#name).to_owned()), self.store.clone_ref(py))
                            }
                        }
                    });
                        quote! { #(#handle_accesses)* }
                    }
                    Fields::Unnamed(fields) => {
                        let handle_accesses = fields.unnamed.iter().enumerate().map(|(num, f)| {
                        let ty = &f.ty;

                        let fn_name = quote::format_ident!("elm{}_py", num);
                        let inside_py_name_str = format!("elm{}", num);

                        quote! {
                            #[pyo3(name = #inside_py_name_str)]
                            pub fn #fn_name(&self, py: pyo3::Python) -> pyo3::PyObject {
                                <#ty as state_link::py::PyState>::build_handle(py, self.inner.inner.index(#num), self.store.clone_ref(py))
                            }
                        }
                    });

                        quote! { #(#handle_accesses)* }
                    }
                    Fields::Unit => todo!(),
                };
                quote! {
                    // Decorating with #[pyo3::pymethods] does not work because there can only be one
                    // pymethods block, at least without adding the multiple-pymethods feature, which has
                    // its own issues: https://github.com/PyO3/pyo3/issues/341
                    impl #impl_generics #name #ty_generics #where_clause {
                        fn store_py(&self, py: pyo3::Python, store: pyo3::Py<::state_link::py::Store>) -> pyo3::PyObject {
                            let node = store.borrow_mut(py).inner.store(self).inner;
                            <Self as ::state_link::py::PyState>::build_handle(py, node, store)
                        }
                    }

                    #[allow(non_camel_case_types)]
                    #[pyo3::pyclass]
                    struct #node_handle_name {
                        inner: #inner_node_handle_name,
                        store: pyo3::Py<state_link::py::Store>,
                    }

                    #[pyo3::pymethods]
                    impl #node_handle_name {
                        #[pyo3(name = "write")]
                        fn write_py(&self, py: pyo3::Python, val: &#name) -> pyo3::PyResult<()> {
                            self.store.borrow_mut(py).inner.write(&self.inner, &val).map_err(state_link::py::map_link_err)
                        }

                        #[pyo3(name = "load")]
                        fn load_py(&self, py: pyo3::Python) -> pyo3::PyObject {
                            self.store.borrow_mut(py).inner.load(&self.inner).into_py(py)
                        }

                        #[pyo3(name = "link_to")]
                        fn link_to_py(&self, py: pyo3::Python, dst: &Self) -> pyo3::PyResult<()> {
                            self.store.borrow_mut(py).inner.link(&self.inner, &dst.inner).map_err(state_link::py::map_link_err)
                        }

                        #[pyo3(name = "mutate")]
                        fn mutate_py(&self, py: pyo3::Python, f: &pyo3::Bound<pyo3::types::PyFunction>) -> pyo3::PyResult<()> {
                            let val_py = self.load_py(py);
                            f.call1((&val_py,))?;
                            let val = val_py.extract::<#name>(py)?;
                            self.write_py(py, &val)
                        }

                        #[pyo3(name = "map")]
                        fn map_py(&self, py: pyo3::Python, f: &pyo3::Bound<pyo3::types::PyFunction>) -> pyo3::PyResult<()> {
                            let val_py = self.load_py(py);
                            let res_py = f.call1((&val_py,))?;
                            let val = res_py.extract::<#name>()?;
                            self.write_py(py, &val)
                        }

                        #handle_access
                    }

                    impl #impl_generics ::state_link::py::PyState for #name #ty_generics #where_clause {
                        fn build_handle(py: pyo3::Python, inner: ::state_link::GenericNodeHandle, store: Py<state_link::py::Store>) -> pyo3::PyObject {
                            use pyo3::ToPyObject;

                            let inner = <<Self as state_link::State>::NodeHandle as state_link::NodeHandle>::pack(inner);
                            let init = #node_handle_name {
                                inner,
                                store,
                            };

                            use pyo3::IntoPyObjectExt;
                            init.into_py_any(py).unwrap()
                        }
                    }
                }
            }
            Data::Enum(_) => {
                //Nothing to do, no handles to give access to
                quote! {
                    impl #impl_generics ::state_link::py::PyState for #name #ty_generics #where_clause {
                        fn build_handle(py: pyo3::Python, inner: ::state_link::GenericNodeHandle, store: Py<state_link::py::Store>) -> pyo3::PyObject {
                            let init = state_link::py::NodeHandleString {
                                inner: <<String as state_link::State>::NodeHandle as state_link::NodeHandle>::pack(inner),
                                store,
                            };

                            use pyo3::IntoPyObjectExt;
                            init.into_py_any(py).unwrap()
                        }
                    }
                }
            }
            Data::Union(_) => todo!(),
        }
    } else {
        quote! {}
    };

    let output = quote! {
        #core_output

        #python_code
    };

    output.into()
}

fn add_trait_bounds(mut generics: Generics) -> Generics {
    for param in &mut generics.params {
        if let GenericParam::Type(ref mut type_param) = *param {
            type_param.bounds.push(parse_quote!(::state_link::State));
        }
    }
    generics
}

fn add_trait_bounds_py(mut generics: Generics) -> Generics {
    for param in &mut generics.params {
        if let GenericParam::Type(ref mut type_param) = *param {
            type_param
                .bounds
                .push(parse_quote!(::state_link::py::PyState));
        }
    }
    generics
}
