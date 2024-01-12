use std::path::PathBuf;

use clap::Parser;
use layout::{
    core::{base::Orientation, geometry::Point, style::StyleAttr},
    std_shapes::shapes::{Arrow, Element, ShapeKind},
};
use macroquad::prelude::*;

fn draw_node(mut pos_x: f32, mut pos_y: f32, content: &str) {
    let font_size = 100;
    let font_scale = 1.0;
    let font = None;

    let center = get_text_center(content, font, font_size, font_scale, 0.0);
    pos_x -= center.x;
    pos_y -= center.y;
    let size = measure_text(content, font, font_size, font_scale);

    draw_rectangle_lines(
        pos_x,
        pos_y - size.offset_y,
        size.width,
        size.height,
        2.0,
        BLACK,
    );
    let params = TextParams {
        font,
        font_size,
        font_scale,
        font_scale_aspect: 1.0,
        rotation: 0.0,
        color: BLACK,
    };
    draw_text_ex(content, pos_x, pos_y, params);
}

#[derive(Parser)]
struct CliArgs {
    #[arg()]
    input: PathBuf,
}

fn to_mq(p: layout::core::geometry::Point) -> Vec2 {
    Vec2::new(p.x as f32, p.y as f32)
}

struct Render;

impl layout::core::format::RenderBackend for Render {
    fn draw_rect(
        &mut self,
        xy: layout::core::geometry::Point,
        size: layout::core::geometry::Point,
        _look: &layout::core::style::StyleAttr,
        _clip: Option<layout::core::format::ClipHandle>,
    ) {
        let size = to_mq(size);
        let mut xy = to_mq(xy);
        xy -= size * 0.5;

        draw_rectangle_lines(xy.x, xy.y, size.x, size.y, 2.0, BLACK);
        //draw_rectangle(xy.x as _, xy.y as _, size.x as _, size.y as _, BLACK);
    }

    fn draw_line(
        &mut self,
        start: layout::core::geometry::Point,
        stop: layout::core::geometry::Point,
        _look: &layout::core::style::StyleAttr,
    ) {
        draw_line(
            start.x as _,
            start.y as _,
            stop.x as _,
            stop.y as _,
            2.0,
            BLACK,
        );
    }

    fn draw_circle(
        &mut self,
        xy: layout::core::geometry::Point,
        size: layout::core::geometry::Point,
        _look: &layout::core::style::StyleAttr,
    ) {
        let size = to_mq(size);
        let mut xy = to_mq(xy);
        xy -= size * 0.5;

        draw_rectangle_lines(xy.x, xy.y, size.x, size.y, 2.0, BLACK);
        //draw_circle_lines(xy.x as _, xy.y as _, size.x as _, 2.0, BLACK);
    }

    fn draw_text(
        &mut self,
        xy: layout::core::geometry::Point,
        text: &str,
        look: &layout::core::style::StyleAttr,
    ) {
        let xy = to_mq(xy);
        let font_size = look.font_size as _;
        let font_scale = 1.0;
        let font = None;

        let params = TextParams {
            font,
            font_size,
            font_scale,
            font_scale_aspect: 1.0,
            rotation: 0.0,
            color: BLACK,
        };
        let size = measure_text(text, font, font_size, font_scale);
        draw_text_ex(
            text,
            xy.x as f32 - size.width * 0.5,
            xy.y as f32 - size.height * 0.5 + size.offset_y,
            params,
        );
    }

    fn draw_arrow(
        &mut self,
        path: &[(layout::core::geometry::Point, layout::core::geometry::Point)],
        _dashed: bool,
        _head: (bool, bool),
        _look: &layout::core::style::StyleAttr,
        _text: &str,
    ) {
        // First the first point, position and control point are swapped:
        let s = path[0].0;
        let e = path[1].1;
        draw_line(s.x as _, s.y as _, e.x as _, e.y as _, 2.0, BLACK);

        for l in path[1..].windows(2) {
            let s = l[0].1;
            let e = l[1].1;
            draw_line(s.x as _, s.y as _, e.x as _, e.y as _, 2.0, BLACK);
        }
    }

    fn create_clip(
        &mut self,
        _xy: layout::core::geometry::Point,
        _size: layout::core::geometry::Point,
        _rounded_px: usize,
    ) -> layout::core::format::ClipHandle {
        todo!()
    }
}

#[macroquad::main("BasicShapes")]
async fn main() {
    let options = CliArgs::parse();

    let contents = std::fs::read_to_string(&options.input).unwrap();
    let mut parser = layout::gv::DotParser::new(&contents);

    let g = match parser.process() {
        Ok(g) => g,
        Err(_err) => {
            parser.print_error();
            panic!("oi no");
        }
    };

    //let mut vg =
    //    layout::topo::layout::VisualGraph::new(layout::core::base::Orientation::TopToBottom);

    //// Define the node styles:
    //let sp0 = ShapeKind::new_box("one");
    //let sp1 = ShapeKind::new_box("two");
    //let look0 = StyleAttr::simple();
    //let look1 = StyleAttr::simple();
    //let sz = Point::new(100., 100.);
    //// Create the nodes:
    //let node0 = Element::create(sp0, look0, Orientation::LeftToRight, sz);
    //let node1 = Element::create(sp1, look1, Orientation::LeftToRight, sz);

    //// Add the nodes to the graph, and save a handle to each node.
    //let handle0 = vg.add_node(node0);
    //let handle1 = vg.add_node(node1);

    //// Add an edge between the nodes.
    //let arrow = Arrow::simple("123");
    //vg.add_edge(arrow, handle0, handle1);

    let sh = screen_height();
    let sw = screen_width();
    let ssize = Vec2::new(sw, sh);
    let mut camera = Camera2D::from_display_rect(Rect::new(0.0, sh, sw, -sh));
    let mut actual_zoom = 1.0;

    let mut gb = layout::gv::GraphBuilder::new();
    gb.visit_graph(&g);

    loop {
        let md = mouse_delta_position() * ssize * 0.5 / actual_zoom;
        if is_mouse_button_down(MouseButton::Left) {
            camera.target += md;
        }
        let zoom_factor = (mouse_wheel().1 * 0.05).exp();
        camera.zoom *= zoom_factor;
        actual_zoom *= zoom_factor;

        set_camera(&camera);

        let mut vg = gb.get();

        clear_background(WHITE);

        vg.do_it(false, false, false, &mut Render);

        next_frame().await
    }
}
