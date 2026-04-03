fn main() {
    // You can optionally experiment here.
}

#[cfg(test)]
mod tests {
    #[test]
    fn slice_out_of_array() {
        let mut a = [1, 2, 3, 4, 5];

        {
            let b = &mut a[1..4];
            b[1] = 7;
        }

        a[3] = 9;

        assert_eq!([1, 2, 7, 9, 5], a);
    }
}
