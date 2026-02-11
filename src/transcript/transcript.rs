pub trait Transcript<T> {
    fn read(&mut self) -> T;
    fn write(&mut self, value: T);
}
