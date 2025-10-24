use ark_ff::Field;
use ark_serialize::Compress;
use ark_std::{
    fs::{self, File},
    io::{BufWriter, Write},
    marker::PhantomData,
    path::Path,
};
use memmap2::Mmap;

use crate::streams::Stream;

#[derive(Debug)]
pub struct FileStream<F: Field> {
    num_variables: usize,
    path: String,
    s: Mmap,
    size_of_serialized: usize,
    _field: PhantomData<F>,
}

impl<F: Field> Clone for FileStream<F> {
    fn clone(&self) -> Self {
        Self::new(self.path.clone())
    }
}

impl<F: Field> FileStream<F> {
    pub fn new(path: String) -> Self {
        let file = File::open(Path::new(&path)).unwrap();
        let mmap = unsafe { Mmap::map(&file) }.unwrap();
        let size_of_serialized = F::ONE.serialized_size(Compress::No);
        let len = mmap.len() / size_of_serialized;
        assert!(len.is_power_of_two());

        let num_variables = len.ilog2() as usize;
        Self {
            num_variables,
            path,
            s: mmap,
            size_of_serialized,
            _field: PhantomData,
        }
    }
    pub fn read_point(mmap: &Mmap, point: usize, size_of_serialized: usize) -> F {
        let offset = point * size_of_serialized;
        let bytes = &mmap[offset..offset + size_of_serialized];
        F::deserialize_uncompressed(bytes).unwrap()
    }
    pub fn write_to_file(path: String, data: &Vec<F>) {
        let file = File::create(path).unwrap();
        let mut writer = BufWriter::new(file);

        for element in data {
            let mut buffer = Vec::new();
            element.serialize_uncompressed(&mut buffer).unwrap();
            writer.write_all(&buffer).unwrap();
        }

        writer.flush().unwrap();
    }
    pub fn delete_file(path: String) {
        fs::remove_file(&path).unwrap();
    }
}

impl<F: Field> Stream<F> for FileStream<F> {
    fn evaluation(&self, point: usize) -> F {
        Self::read_point(&self.s, point, self.size_of_serialized)
    }

    fn num_variables(&self) -> usize {
        self.num_variables
    }
}

#[cfg(test)]
mod tests {
    // use crate::{
    //     multilinear_product::BlendyProductProver,
    //     prover::{ProductProverConfig, Prover},
    //     streams::{stream::multivariate_claim, FileStream, GraycodeOrder, MemoryStream, Stream},
    //     tests::F19,
    //     ProductSumcheck,
    // };

    // #[test]
    // fn consistency_with_memory_stream() {
    //     // create and write to the file we'll stream from
    //     let path = "file_stream_consistency_with_memory_stream_test_evals.bin".to_string();
    //     let evals: Vec<F19> = vec![
    //         F19::from(0),
    //         F19::from(1),
    //         F19::from(2),
    //         F19::from(3),
    //         F19::from(4),
    //         F19::from(5),
    //         F19::from(6),
    //         F19::from(7),
    //     ];
    //     FileStream::<F19>::write_to_file(path.clone(), &evals);

    //     // instantiate the file stream
    //     let s_file: FileStream<F19> = FileStream::new(path.clone());
    //     let claim: F19 = multivariate_claim(s_file.clone());

    //     // prove over it using BlendyProver
    //     let mut blendy_prover_file_stream =
    //         BlendyProductProver::<F19, FileStream<F19>, GraycodeOrder>::new(<BlendyProductProver<
    //             F19,
    //             FileStream<F19>,
    //             GraycodeOrder,
    //         > as Prover<F19>>::ProverConfig::default(
    //             claim,
    //             s_file.num_variables(),
    //             vec![s_file.clone(), s_file],
    //         ));
    //     let blendy_prover_file_stream_transcript =
    //         ProductSumcheck::<F19>::prove::<
    //             FileStream<F19>,
    //             BlendyProductProver<F19, FileStream<F19>, GraycodeOrder>,
    //         >(&mut blendy_prover_file_stream, &mut ark_std::test_rng());

    //     // instantiate the memory stream
    //     let s_memory: MemoryStream<F19> = MemoryStream::new(evals);
    //     let claim: F19 = multivariate_claim(s_memory.clone());

    //     // prove over it using BlendyProver
    //     let mut blendy_prover_memory_stream = BlendyProductProver::<
    //         F19,
    //         MemoryStream<F19>,
    //         GraycodeOrder,
    //     >::new(<BlendyProductProver<
    //         F19,
    //         MemoryStream<F19>,
    //         GraycodeOrder,
    //     > as Prover<F19>>::ProverConfig::default(
    //         claim,
    //         s_memory.num_variables(),
    //         vec![s_memory.clone(), s_memory],
    //     ));
    //     let blendy_prover_memory_stream_transcript =
    //         ProductSumcheck::<F19>::prove::<
    //             MemoryStream<F19>,
    //             BlendyProductProver<F19, MemoryStream<F19>, GraycodeOrder>,
    //         >(&mut blendy_prover_memory_stream, &mut ark_std::test_rng());

    //     // cleanup
    //     FileStream::<F19>::delete_file(path);

    //     // Assert they computed the same thing
    //     assert_eq!(
    //         blendy_prover_file_stream_transcript.prover_messages,
    //         blendy_prover_memory_stream_transcript.prover_messages
    //     );
    // }
}
