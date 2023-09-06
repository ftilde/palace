use proc_macro::{self, TokenStream};
use quote::quote;
use syn::{parse_macro_input, parse_quote, Data, DeriveInput, Fields, GenericParam, Generics};

#[proc_macro_derive(State)]
pub fn my_macro_here_derive(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);

    // Used in the quasi-quotation below as `#name`.
    let name = input.ident;

    let visibility = input.vis;

    // Add a bound `T: HeapSize` to every type parameter T.
    let generics = add_trait_bounds(input.generics);
    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();

    let data = input.data;

    let (store, load, write, handle_access) = match data {
        Data::Struct(s) => match s.fields {
            Fields::Named(fields) => {
                let field_load = fields.named.iter().map(|f| {
                    let name = &f.ident;
                    let ty = &f.ty;
                    quote! {
                        #name: {
                            let field_name = stringify!(#name);
                            let loc = map.get(field_name).ok_or(state_link::Error::MissingField(field_name.to_owned()))?;
                            <#ty>::load(store, *loc)?
                        },
                    }
                });

                let load_code = quote! {
                    if let state_link::ResolveResult::Struct(map) = store.to_val(location) {
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
                        map.insert(stringify!(#name).to_owned(), self.#name.store(store));
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
                            let loc = map.get(field_name).ok_or(state_link::Error::MissingField(field_name.to_owned()))?;
                            self.#name.write(store, *loc)?;
                        }
                    }
                });

                let write_code = quote! {
                    let mut map = if let ::state_link::ResolveResult::Struct(map) = store.to_val(at) {
                        map.clone()
                    } else {
                        return Err(::state_link::Error::IncorrectType);
                    };

                    #(#field_write)*;

                    store.write_at(::state_link::Node::Dir(map), at);
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
                    if let state_link::ResolveResult::Seq(seq) = store.to_val(location) {
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
                        seq.push(self.#num.store(store));
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
                            self.#num.write(store, *loc)?;
                        }
                    }
                });

                let num_fields = fields.unnamed.len();

                let write_code = quote! {
                    let mut seq = if let ::state_link::ResolveResult::Seq(seq) = store.to_val(at) {
                        seq.clone()
                    } else {
                        return Err(::state_link::Error::IncorrectType);
                    };

                    if seq.len() != #num_fields {
                        return Err(::state_link::Error::IncorrectType);
                    }

                    #(#field_write)*;

                    store.write_at(::state_link::Node::Seq(seq), at);
                    Ok(())
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
                    if let state_link::ResolveResult::Atom(state_link::Value::Unit) = store.to_val(location) {
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
        Data::Enum(_) => unimplemented!(),
        Data::Union(_) => unimplemented!(),
    };

    let node_handle_name = quote::format_ident!("__NodeHandle_{}", name);

    let output = quote! {
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
