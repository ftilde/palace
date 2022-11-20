fn integer_division_rounded_up( a: u64, b: u64 ) -> u64 {
    (a + b - 1) / b
}

trait Array : Sized {
    fn size(&self) -> u64;
    fn brick_size(&self) -> u32;
    fn brick_count(&self) -> u32;

    fn as_image(&self) -> ArrayAsImage<Self> {
        ArrayAsImage { array: self }
    }
    fn as_volume(&self) -> ArrayAsVolume<Self> {
        ArrayAsVolume { array: self }
    }
}
trait Image : Sized {
    fn dims(&self) -> cgmath::Vector2<u32>;
    fn brick_dims(&self) -> cgmath::Vector2<u32>;
    fn dims_in_bricks(&self) -> cgmath::Vector2<u32> {
        let dims = self.dims();
        let brick_dims = self.brick_dims();
        dims.zip(brick_dims, |a: u32, b:u32| integer_division_rounded_up(a as u64, b as u64) as u32)
    }

    fn size(&self) -> u64 {
        let dims = Image::dims(self);
        (dims.x * dims.y) as u64
    }
    fn brick_size(&self) -> u32 {
        let brick_dims = Image::brick_dims(self);
        brick_dims.x * brick_dims.y
    }
    fn brick_count(&self) -> u32 {
        let dims_in_bricks = Image::dims_in_bricks(self);
        dims_in_bricks.x * dims_in_bricks.y
    }

    fn pixel_to_brick(&self, pixel: cgmath::Vector2<u32>) -> cgmath::Vector2<u32> {
        pixel.zip(self.brick_dims(), |a: u32, b:u32| a / b)
    }
    fn brick_to_pixel_begin(&self, brick: cgmath::Vector2<u32>) -> cgmath::Vector2<u32> {
        brick.zip(self.brick_dims(), |a: u32, b:u32| a * b)
    }
    fn brick_to_pixel_end(&self, brick: cgmath::Vector2<u32>) -> cgmath::Vector2<u32> {
        self.brick_to_pixel_begin(brick + cgmath::Vector2::<u32>::new(1, 1))
    }

    fn as_array(&self) -> ImageAsArray<Self> {
        ImageAsArray { image: self }
    }
    fn as_volume(&self) -> ImageAsVolume<Self> {
        ImageAsVolume { image: self }
    }
}
trait Volume : Sized {
    fn dims(&self) -> cgmath::Vector3<u32>;
    fn brick_dims(&self) -> cgmath::Vector3<u32>;
    fn dims_in_bricks(&self) -> cgmath::Vector3<u32> {
        let dims = self.dims();
        let brick_dims = self.brick_dims();
        dims.zip(brick_dims, |a: u32, b:u32| integer_division_rounded_up(a as u64, b as u64) as u32)
    }

    fn size(&self) -> u64 {
        let dims = self.dims();
        (dims.x as u64) * (dims.y as u64) * (dims.z as u64)
    }
    fn brick_size(&self) -> u32 {
        let brick_dims = self.brick_dims();
        brick_dims.x * brick_dims.y * brick_dims.z
    }
    fn brick_count(&self) -> u32 {
        let dims_in_bricks = self.dims_in_bricks();
        dims_in_bricks.x * dims_in_bricks.y * dims_in_bricks.z
    }

    fn voxel_to_brick(&self, voxel: cgmath::Vector3<u32>) -> cgmath::Vector3<u32> {
        voxel.zip(self.brick_dims(), |a: u32, b:u32| a / b)
    }
    fn brick_to_voxel_begin(&self, brick: cgmath::Vector3<u32>) -> cgmath::Vector3<u32> {
        brick.zip(self.brick_dims(), |a: u32, b:u32| a * b)
    }
    fn brick_to_voxel_end(&self, brick: cgmath::Vector3<u32>) -> cgmath::Vector3<u32> {
        self.brick_to_voxel_begin(brick + cgmath::Vector3::<u32>::new(1, 1, 1))
    }

    fn as_array(&self) -> VolumeAsArray<Self> {
        VolumeAsArray { volume: self }
    }
}



struct ArrayMetaData {
    size: u64,
    brick_size: u32
}
impl Array for ArrayMetaData {
    fn size(&self) -> u64 {
        self.size
    }
    fn brick_size(&self) -> u32 {
        self.brick_size
    }
    fn brick_count(&self) -> u32 {
        integer_division_rounded_up(self.size(), self.brick_size() as u64) as u32
    }
}

struct ImageMetaData {
    dims: cgmath::Vector2<u32>,
    brick_dims: cgmath::Vector2<u32>
}
impl Image for ImageMetaData {
    fn dims(&self) -> cgmath::Vector2<u32> {
        self.dims
    }
    fn brick_dims(&self) -> cgmath::Vector2<u32> {
        self.brick_dims
    }
}

struct VolumeMetaData {
    dims: cgmath::Vector3<u32>,
    brick_dims: cgmath::Vector3<u32>
}
impl Volume for VolumeMetaData {
    fn dims(&self) -> cgmath::Vector3<u32> {
        self.dims
    }
    fn brick_dims(&self) -> cgmath::Vector3<u32> {
        self.brick_dims
    }
}


struct ArrayAsImage<'a, T: Array> {
    array: &'a T
}
impl<'a, T: Array> Image for ArrayAsImage<'a, T> {
    fn dims(&self) -> cgmath::Vector2<u32> {
        let size = self.array.size();
        cgmath::Vector2::<u32>::new(1, size as u32)
    }
    fn brick_dims(&self) -> cgmath::Vector2<u32> {
        let brick_size = self.array.brick_size();
        cgmath::Vector2::<u32>::new(1, brick_size)
    }
}
struct ArrayAsVolume<'a, T: Array> {
    array: &'a T
}
impl<'a, T: Array> Volume for ArrayAsVolume<'a, T> {
    fn dims(&self) -> cgmath::Vector3<u32> {
        let size = self.array.size();
        cgmath::Vector3::<u32>::new(1, 1, size as u32)
    }
    fn brick_dims(&self) -> cgmath::Vector3<u32> {
        let brick_size = self.array.brick_size();
        cgmath::Vector3::<u32>::new(1, 1, brick_size)
    }
}
struct ImageAsArray<'a, T: Image> {
    image: &'a T
}
impl<'a, T: Image> Array for ImageAsArray<'a, T> {
    fn size(&self) -> u64 {
        self.image.size()
    }
    fn brick_size(&self) -> u32 {
        self.image.brick_size()
    }
    fn brick_count(&self) -> u32 {
        self.image.brick_count()
    }
}
struct ImageAsVolume<'a, T: Image> {
    image: &'a T
}
impl<'a, T: Image> Volume for ImageAsVolume<'a, T> {
    fn dims(&self) -> cgmath::Vector3<u32> {
        let dims = self.image.dims();
        cgmath::Vector3::<u32>::new(1, dims.x, dims.y)
    }
    fn brick_dims(&self) -> cgmath::Vector3<u32> {
        let brick_dims = self.image.brick_dims();
        cgmath::Vector3::<u32>::new(1, brick_dims.x, brick_dims.y)
    }
}

struct VolumeAsArray<'a, T: Volume> {
    volume: &'a T
}
impl<'a, T: Volume> Array for VolumeAsArray<'a, T> {
    fn size(&self) -> u64 {
        self.volume.size()
    }
    fn brick_size(&self) -> u32 {
        self.volume.brick_size()
    }
    fn brick_count(&self) -> u32 {
        self.volume.brick_count()
    }
}