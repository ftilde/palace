pub fn div_round_up<
    U: Copy + std::ops::Add<Output = U> + std::ops::Div<Output = U> + std::ops::Sub<Output = U>,
>(
    v1: U,
    v2: U,
) -> U {
    (v1 + v2 - (v2 / v2)) / v2
}

// Compute the size of an element in an array, i.e. including the padding to the next element.
pub const fn array_elm_size<T>() -> usize {
    let len = std::mem::size_of::<T>();
    let align = std::mem::align_of::<T>();

    // See https://doc.rust-lang.org/std/alloc/struct.Layout.html#method.pad_to_align
    // (which is not const, yet, for some reason)
    let len_rounded_up = len.wrapping_add(align).wrapping_sub(1) & !align.wrapping_sub(1);
    let padding = len_rounded_up.wrapping_sub(len);
    len + padding
}

pub fn num_elms_in_array<T>(size_in_bytes: usize) -> usize {
    let size_with_padding = crate::util::array_elm_size::<T>();
    // TODO: This may still break if the array size does not include
    // padding for the last element, but it probably should. See
    // https://rust-lang.github.io/unsafe-code-guidelines/layout/arrays-and-slices.html
    let num_elements = size_in_bytes / size_with_padding;
    num_elements
}

//pub type Map<K, V> = std::collections::BTreeMap<K, V>;
pub type Map<K, V> = ahash::HashMap<K, V>;
pub type MapEntry<'a, K, V> = std::collections::hash_map::Entry<'a, K, V>;
//pub type Set<K> = std::collections::BTreeSet<K>;
pub type Set<K> = ahash::HashSet<K>;
