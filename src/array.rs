fn integer_division_rounded_up( a: u64, b: u64 ) -> u64 {
    (a + b - 1) / b
}

trait Array {
    fn size(&self) -> u64;
    fn brick_size(&self) -> u32;
    fn brick_count(&self) -> u32 {
        integer_division_rounded_up(self.size(), self.brick_size() as u64) as u32
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
}



trait Image : Array {
    fn dims(&self) -> cgmath::Vector2<u32>;
    fn brick_dims(&self) -> cgmath::Vector2<u32>;
    fn dims_in_bricks(&self) -> cgmath::Vector2<u32> {
        let dims = self.dims();
        let brick_dims = self.brick_dims();
        dims.zip(brick_dims, |a: u32, b:u32| integer_division_rounded_up(a as u64, b as u64) as u32)
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
}
struct ImageMetaData {
    dims: cgmath::Vector2<u32>,
    brick_dims: cgmath::Vector2<u32>
}
impl Array for ImageMetaData {
    fn size(&self) -> u64 {
        let dims = self.dims();
        (dims.x * dims.y) as u64
    }
    fn brick_size(&self) -> u32 {
        let brick_dims = self.brick_dims();
        brick_dims.x * brick_dims.y
    }
    fn brick_count(&self) -> u32 {
        let dims_in_bricks = self.dims_in_bricks();
        dims_in_bricks.x * dims_in_bricks.y
    }
}
impl Image for ImageMetaData {
    fn dims(&self) -> cgmath::Vector2<u32> {
        self.dims
    }
    fn brick_dims(&self) -> cgmath::Vector2<u32> {
        self.brick_dims
    }
}



trait Volume : Array {
    fn dims(&self) -> cgmath::Vector3<u32>;
    fn brick_dims(&self) -> cgmath::Vector3<u32>;
    fn dims_in_bricks(&self) -> cgmath::Vector3<u32> {
        let dims = self.dims();
        let brick_dims = self.brick_dims();
        dims.zip(brick_dims, |a: u32, b:u32| integer_division_rounded_up(a as u64, b as u64) as u32)
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
}
struct VolumeMetaData {
    dims: cgmath::Vector3<u32>,
    brick_dims: cgmath::Vector3<u32>
}
impl Array for VolumeMetaData {
    fn size(&self) -> u64 {
        let dims = self.dims();
        (dims.x * dims.y * dims.z) as u64
    }
    fn brick_size(&self) -> u32 {
        let brick_dims = self.brick_dims();
        brick_dims.x * brick_dims.y * brick_dims.z
    }
    fn brick_count(&self) -> u32 {
        let dims_in_bricks = self.dims_in_bricks();
        dims_in_bricks.x * dims_in_bricks.y * dims_in_bricks.z
    }
}
impl Volume for VolumeMetaData {
    fn dims(&self) -> cgmath::Vector3<u32> {
        self.dims
    }
    fn brick_dims(&self) -> cgmath::Vector3<u32> {
        self.brick_dims
    }
}