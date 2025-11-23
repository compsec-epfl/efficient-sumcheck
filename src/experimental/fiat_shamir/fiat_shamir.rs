pub trait FiatShamir<T> {
    fn absorb(&mut self, value: T);
    fn squeeze(&mut self) -> T;
}
