use std::collections::HashMap;

pub use state_link_derive::*;

#[cfg(feature = "python")]
pub mod py;

pub type Map = HashMap<String, NodeRef>;

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct NodeRef {
    index: usize,
}

#[derive(Default, Debug)]
pub struct Store {
    elms: Vec<Node>,
}

#[derive(Debug)]
pub enum Node {
    Dir(Map),
    Seq(Vec<NodeRef>),
    Val(Value),
    Link(NodeRef),
}

pub enum ResolveResult<'a> {
    Struct(&'a Map),
    Seq(&'a Vec<NodeRef>),
    Atom(Value),
}

#[derive(Debug, Copy, Clone)]
pub enum Error {
    IncorrectType,
    SeqTooShort,
    MissingField(&'static str),
    LinkReferenceCycle,
    UnknownVariant,
}

pub type Result<S> = std::result::Result<S, Error>;

pub trait State: Sized {
    type NodeHandle: NodeHandle<NodeType = Self>;
    fn write(&self, store: &mut Store, at: NodeRef) -> Result<()>;
    fn store(&self, store: &mut Store) -> NodeRef;
    fn load(store: &Store, location: NodeRef) -> Result<Self>;
}

impl State for f32 {
    type NodeHandle = NodeHandleSpecialized<Self>;
    fn store(&self, store: &mut Store) -> NodeRef {
        store.push(Node::Val(Value::F32(*self)))
    }

    fn load(store: &Store, location: NodeRef) -> Result<Self> {
        if let ResolveResult::Atom(Value::F32(v)) = store.to_val(location)? {
            Ok(v)
        } else {
            Err(Error::IncorrectType)
        }
    }

    fn write(&self, store: &mut Store, at: NodeRef) -> Result<()> {
        store.write_at(Node::Val(Value::F32(*self)), at)
    }
}

impl State for u32 {
    type NodeHandle = NodeHandleSpecialized<Self>;

    fn store(&self, store: &mut Store) -> NodeRef {
        store.push(Node::Val(Value::U32(*self)))
    }

    fn load(store: &Store, location: NodeRef) -> Result<Self> {
        if let ResolveResult::Atom(Value::U32(v)) = store.to_val(location)? {
            Ok(v)
        } else {
            Err(Error::IncorrectType)
        }
    }
    fn write(&self, store: &mut Store, at: NodeRef) -> Result<()> {
        store.write_at(Node::Val(Value::U32(*self)), at)
    }
}

impl State for String {
    type NodeHandle = NodeHandleSpecialized<Self>;

    fn store(&self, store: &mut Store) -> NodeRef {
        store.push(Node::Val(Value::String(self.clone())))
    }

    fn load(store: &Store, location: NodeRef) -> Result<Self> {
        if let ResolveResult::Atom(Value::String(v)) = store.to_val(location)? {
            Ok(v)
        } else {
            Err(Error::IncorrectType)
        }
    }
    fn write(&self, store: &mut Store, at: NodeRef) -> Result<()> {
        store.write_at(Node::Val(Value::String(self.clone())), at)
    }
}

impl<V: State> State for Vec<V> {
    type NodeHandle = NodeHandleSpecialized<Self>;
    fn store(&self, store: &mut Store) -> NodeRef {
        let refs = self.iter().map(|v| v.store(store)).collect();
        store.push(Node::Seq(refs))
    }

    fn load(store: &Store, location: NodeRef) -> Result<Self> {
        if let ResolveResult::Seq(seq) = store.to_val(location)? {
            seq.into_iter()
                .map(|loc| V::load(store, *loc))
                .collect::<std::result::Result<Vec<V>, _>>()
        } else {
            Err(Error::IncorrectType)
        }
    }

    fn write(&self, store: &mut Store, at: NodeRef) -> Result<()> {
        let mut seq = if let ResolveResult::Seq(seq) = store.to_val(at)? {
            seq.clone() //TODO: instead of cloning we can probably also just take the old value out
        } else {
            return Err(Error::IncorrectType);
        };

        let min_len = self.len().min(seq.len());
        for (v, slot) in self[..min_len].iter().zip(seq[..min_len].iter()) {
            v.write(store, *slot)?;
        }

        match self.len().cmp(&seq.len()) {
            std::cmp::Ordering::Less => {
                //TODO: delete other elements somehow?
            }
            std::cmp::Ordering::Equal => {
                // Nothing to do
            }
            std::cmp::Ordering::Greater => {
                for v in &self[min_len..] {
                    seq.push(v.store(store));
                }
            }
        }

        store.write_at(Node::Seq(seq), at)
    }
}

impl<V: State> NodeHandleSpecialized<Vec<V>> {
    pub fn at(&self, i: usize) -> <V as State>::NodeHandle {
        NodeHandle::pack(self.inner.index(i))
    }
}

impl<const I: usize, V: State> State for [V; I] {
    type NodeHandle = NodeHandleSpecialized<Self>;
    fn store(&self, store: &mut Store) -> NodeRef {
        let refs = self.iter().map(|v| v.store(store)).collect();
        store.push(Node::Seq(refs))
    }

    fn load(store: &Store, location: NodeRef) -> Result<Self> {
        // If only std::array::try_from_fn were stable...
        // TODO: replace once it is https://github.com/rust-lang/rust/issues/89379
        if let ResolveResult::Seq(seq) = store.to_val(location)? {
            let results: [Result<V>; I] = std::array::from_fn(|i| {
                seq.get(i)
                    .ok_or(Error::SeqTooShort)
                    .and_then(|v| V::load(store, *v))
            });
            for (i, r) in results.iter().enumerate() {
                if let Err(_e) = r {
                    match results.into_iter().nth(i).unwrap() {
                        Ok(_) => std::unreachable!(),
                        Err(e) => return Err(e),
                    }
                }
            }
            Ok(results.map(|v| match v {
                Ok(e) => e,
                Err(_) => std::unreachable!(),
            }))
        } else {
            Err(Error::IncorrectType)
        }
    }

    fn write(&self, store: &mut Store, at: NodeRef) -> Result<()> {
        let seq = if let ResolveResult::Seq(seq) = store.to_val(at)? {
            seq.clone() //TODO: instead of cloning we can probably also just take the old value out
        } else {
            return Err(Error::IncorrectType);
        };

        if seq.len() != I {
            return Err(Error::IncorrectType);
        }

        for (v, slot) in self.iter().zip(seq.iter()) {
            v.write(store, *slot)?;
        }

        Ok(())
    }
}

impl<const I: usize, V: State> NodeHandleSpecialized<[V; I]> {
    pub fn at(&self, i: usize) -> <V as State>::NodeHandle {
        NodeHandle::pack(self.inner.index(i))
    }
}

impl<T> State for std::marker::PhantomData<T> {
    type NodeHandle = NodeHandleSpecialized<Self>;

    fn store(&self, store: &mut Store) -> NodeRef {
        store.push(Node::Val(Value::Unit))
    }

    fn load(store: &Store, location: NodeRef) -> Result<Self> {
        if let ResolveResult::Atom(Value::Unit) = store.to_val(location)? {
            Ok(Default::default())
        } else {
            Err(Error::IncorrectType)
        }
    }
    fn write(&self, store: &mut Store, at: NodeRef) -> Result<()> {
        store.write_at(Node::Val(Value::Unit), at)
    }
}

const MAX_LINK_FOLLOWS: usize = 32;

impl Store {
    pub fn push(&mut self, val: Node) -> NodeRef {
        let index = self.elms.len();
        self.elms.push(val);
        NodeRef { index }
    }

    fn resolve_links(&self, mut node: NodeRef) -> Result<NodeRef> {
        for _ in 0..MAX_LINK_FOLLOWS {
            if let Node::Link(l) = &self.elms[node.index] {
                node = *l
            } else {
                return Ok(node);
            }
        }
        Err(Error::LinkReferenceCycle)
    }

    pub fn write_at(&mut self, val: Node, at: NodeRef) -> Result<()> {
        let at = self.resolve_links(at)?;
        self.elms[at.index] = val;
        Ok(())
    }

    pub fn store<T: State>(&mut self, v: &T) -> T::NodeHandle {
        T::NodeHandle::pack(GenericNodeHandle::new_at(v.store(self)))
    }

    pub fn load_unchecked<T: State>(&self, node: &GenericNodeHandle) -> T {
        T::load(self, self.resolve(node).unwrap()).unwrap()
    }
    pub fn load<H: NodeHandle>(&self, node: &H) -> H::NodeType {
        let node = node.unpack();
        self.load_unchecked(&node)
    }

    pub fn link_unchecked(
        &mut self,
        src: &GenericNodeHandle,
        target: &GenericNodeHandle,
    ) -> Result<()> {
        let src_node = self.walk(src.node, &src.path).unwrap();
        let target_node = self.walk(target.node, &target.path).unwrap();

        if src_node == target_node {
            return Err(Error::LinkReferenceCycle);
        }

        self.elms[src_node.index] = Node::Link(target_node);
        Ok(())
    }

    pub fn link<H: NodeHandle>(&mut self, src: &H, target: &H) -> Result<()> {
        let src = src.unpack();
        let target = target.unpack();

        self.link_unchecked(src, target)
    }

    pub fn resolve(&self, loc: &GenericNodeHandle) -> Option<NodeRef> {
        self.walk(loc.node, &loc.path)
    }

    pub fn write_unchecked<T: State>(
        &mut self,
        target: &GenericNodeHandle,
        value: &T,
    ) -> Result<()> {
        let loc = self.resolve(target).unwrap();

        value.write(self, loc)
    }

    pub fn write<H: NodeHandle>(&mut self, target: &H, value: &H::NodeType) -> Result<()> {
        self.write_unchecked(target.unpack(), value)
    }

    fn walk_inner(&self, root: NodeRef, p: &[PathElm], link_budget: usize) -> Option<NodeRef> {
        match p {
            [] => Some(root),
            [current, rest @ ..] => match &self.elms[root.index] {
                Node::Dir(d) => {
                    let e = d.get(current.str()?)?;
                    self.walk(*e, rest)
                }
                Node::Seq(s) => {
                    let e = s.get(current.index()?)?;
                    self.walk(*e, rest)
                }
                Node::Val(_) => None,
                Node::Link(l) if link_budget > 0 => self.walk_inner(*l, p, link_budget - 1),
                Node::Link(_l) => None, //TODO: error instead?
            },
        }
    }

    pub fn walk(&self, root: NodeRef, p: &[PathElm]) -> Option<NodeRef> {
        self.walk_inner(root, p, MAX_LINK_FOLLOWS)
    }

    fn to_val_inner(&self, root: NodeRef, link_budget: usize) -> Result<ResolveResult> {
        match &self.elms[root.index] {
            Node::Dir(s) => Ok(ResolveResult::Struct(&s)),
            Node::Seq(s) => Ok(ResolveResult::Seq(&s)),
            Node::Val(v) => Ok(ResolveResult::Atom(v.clone())),
            Node::Link(l) if link_budget > 0 => self.to_val_inner(*l, link_budget - 1),
            Node::Link(_l) => Err(Error::LinkReferenceCycle),
        }
    }
    pub fn to_val(&self, root: NodeRef) -> Result<ResolveResult> {
        self.to_val_inner(root, MAX_LINK_FOLLOWS)
    }
}

#[derive(derive_more::From, Clone)]
pub enum PathElm {
    Named(String),
    Index(usize),
}

impl PathElm {
    fn str(&self) -> Option<&str> {
        if let PathElm::Named(s) = self {
            Some(s)
        } else {
            None
        }
    }
    fn index(&self) -> Option<usize> {
        if let PathElm::Index(i) = self {
            Some(*i)
        } else {
            None
        }
    }
}

#[derive(Clone, Debug)]
pub enum Value {
    F32(f32),
    U32(u32),
    String(String),
    Unit,
}

pub trait NodeHandle {
    type NodeType: State;

    fn pack(t: GenericNodeHandle) -> Self;
    fn unpack(&self) -> &GenericNodeHandle;
}

#[derive(Clone)]
pub struct GenericNodeHandle {
    node: NodeRef,
    path: Vec<PathElm>,
}

impl GenericNodeHandle {
    fn new_at(node: NodeRef) -> Self {
        Self {
            node,
            path: Vec::new(),
        }
    }
    pub fn index(&self, i: usize) -> GenericNodeHandle {
        GenericNodeHandle {
            node: self.node,
            path: {
                let mut p = self.path.clone();
                p.push(PathElm::Index(i));
                p
            },
        }
    }
    pub fn named(&self, name: String) -> GenericNodeHandle {
        GenericNodeHandle {
            node: self.node,
            path: {
                let mut p = self.path.clone();
                p.push(PathElm::Named(name));
                p
            },
        }
    }
}

pub struct NodeHandleSpecialized<T> {
    inner: GenericNodeHandle,
    _marker: std::marker::PhantomData<T>,
}
impl<T> Clone for NodeHandleSpecialized<T> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            _marker: Default::default(),
        }
    }
}

impl<T: State> NodeHandle for NodeHandleSpecialized<T> {
    type NodeType = T;
    fn pack(inner: GenericNodeHandle) -> Self {
        Self {
            inner,
            _marker: Default::default(),
        }
    }

    fn unpack(&self) -> &GenericNodeHandle {
        &self.inner
    }
}
