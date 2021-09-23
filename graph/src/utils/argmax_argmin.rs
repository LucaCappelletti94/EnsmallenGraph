use rayon::prelude::*;

pub trait ArgMaxArgMin<T> {
    fn argmax(self) -> Option<(usize, T)>;
    fn argmin(self) -> Option<(usize, T)>;
    fn argminmax(self) -> Option<((usize, T), (usize, T))>;
}

impl<I, T: PartialOrd + Copy + Sized + Send> ArgMaxArgMin<T> for I
where
    I: IndexedParallelIterator<Item = T>,
{
    fn argmax(self) -> Option<(usize, T)> {
        self.enumerate().map(|e| Some(e)).reduce(
            || None,
            |a, b| match (a, b) {
                (Some((i, a)), Some((j, b))) => Some(if a > b { (i, a) } else { (j, b) }),
                (Some(a), None) => Some(a),
                (None, Some(b)) => Some(b),
                (None, None) => None,
            },
        )
    }

    fn argmin(self) -> Option<(usize, T)> {
        self.enumerate().map(|e| Some(e)).reduce(
            || None,
            |a, b| match (a, b) {
                (Some((i, a)), Some((j, b))) => Some(if a < b { (i, a) } else { (j, b) }),
                (Some(a), None) => Some(a),
                (None, Some(b)) => Some(b),
                (None, None) => None,
            },
        )
    }

    fn argminmax(self) -> Option<((usize, T), (usize, T))> {
        self.enumerate().map(|e| Some((e, e))).reduce(
            || None,
            |a, b| match (a, b) {
                (Some(((i1, a1), (i2, a2))), Some(((j1, b1), (j2, b2)))) => Some((
                    if a1 < b1 { (i1, a1) } else { (j1, b1) },
                    if a2 > b2 { (i2, a2) } else { (j2, b2) },
                )),
                (Some(a), None) => Some(a),
                (None, Some(b)) => Some(b),
                (None, None) => None,
            },
        )
    }
}
