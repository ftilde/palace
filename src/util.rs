pub fn div_round_up(v1: u32, v2: u32) -> u32 {
    (v1 + v2 - 1) / v2
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
