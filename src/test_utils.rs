macro_rules! assert_delta {
    ($x:expr, $y:expr, $d:expr) => {
        if !($x - $y < $d && $y - $x < $d) {
            panic!(
                "value {} is different than {} by greater than {} delta",
                $x, $y, $d
            );
        }
    };
}

pub(crate) use assert_delta;
