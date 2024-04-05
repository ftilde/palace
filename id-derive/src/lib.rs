use proc_macro::{self, TokenStream};
use quote::quote;
use syn::{parse_macro_input, Data, DeriveInput, Fields, Generics};

#[proc_macro_derive(Identify)]
pub fn derive_state(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);

    // Used in the quasi-quotation below as `#name`.
    let name = input.ident;

    // Add a bound `T: HeapSize` to every type parameter T.
    let generics = add_trait_bounds(input.generics.clone());
    let (impl_generics, ty_generics, _where_clause) = generics.split_for_impl();

    let data = input.data;

    let code = match &data {
        Data::Struct(s) => match &s.fields {
            Fields::Named(fields) => {
                let field_ids = fields.named.iter().map(|f| {
                    let name = &f.ident;
                    quote! {
                        self.#name.id(),
                    }
                });

                quote! {
                    ::id::Id::combine(&[
                        #(#field_ids)*
                    ])
                }
            }
            Fields::Unnamed(fields) => {
                let field_ids = fields.unnamed.iter().enumerate().map(|(num, _f)| {
                    let num = syn::Index::from(num);
                    quote! {
                        self.#num.id(),
                    }
                });
                quote! {
                    ::id::Id::combine(&[
                        #(#field_ids)*
                    ])
                }
            }
            Fields::Unit => {
                quote! {
                    ::id::Id::from_data(&[])
                }
            }
        },
        Data::Enum(e) => {
            let variants = e.variants.iter().enumerate().map(|(i, v)| {
                let name = &v.ident;
                match &v.fields {
                    Fields::Named(f) => {
                        let field_names = f.named.iter().map(|v| &v.ident);
                        let field_ids = field_names.clone().map(|name| {
                            quote! {
                                #name.id()
                            }
                        });
                        quote! {
                            Self::#name { #(#field_names,)* } => {
                                ::id::Id::combine(&[
                                    #i.id(),
                                    #(#field_ids,)*
                                ])
                            }
                        }
                    }
                    Fields::Unnamed(f) => {
                        let field_names = (0..f.unnamed.len()).map(|num| {
                            let name = quote::format_ident!("elm{}", num);
                            quote! { #name }
                        });
                        let field_ids = field_names.clone().map(|name| {
                            quote! {
                                #name.id()
                            }
                        });
                        quote! {
                            Self::#name ( #(#field_names,)* ) => {
                                ::id::Id::combine(&[
                                    #i.id(),
                                    #(#field_ids,)*
                                ])
                            }
                        }
                    }
                    Fields::Unit => quote! {
                        Self::#name => #i.id(),
                    },
                }
            });
            quote! {
                match self {
                    #(#variants)*
                }
            }
        }
        Data::Union(_) => unimplemented!(),
    };

    let output = quote! {
        impl #impl_generics ::id::Identify for #name #ty_generics {
            fn id(&self) -> ::id::Id {
                #code
            }
        }
    };

    output.into()
}

fn add_trait_bounds(generics: Generics) -> Generics {
    //for param in &mut generics.params {
    //    if let GenericParam::Type(ref mut type_param) = *param {
    //        type_param.bounds.push(parse_quote!(::state_link::State));
    //    }
    //}
    generics
}
