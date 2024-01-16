use std::{
    collections::{HashMap, HashSet},
    path::PathBuf,
};

use clap::Parser;
use gs_core::*;
use layout::{
    core::{base::Orientation, geometry::Point, style::StyleAttr},
    std_shapes::shapes::{Arrow, Element, ShapeKind},
    topo::layout::VisualGraph,
};
use macroquad::prelude::*;

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
        let mut size = to_mq(size);
        let mut xy = to_mq(xy);
        xy -= size * 0.5;
        size *= 2.0;

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

#[derive(Default)]
struct Graph {
    nodes: HashMap<u64, String>,
    edges: HashSet<Edge>,
}

impl Graph {
    fn apply(&mut self, e: Event) {
        match e {
            Event::AddNode(n) => assert!(
                self.nodes.insert(n.id, n.label.clone()).is_none(),
                "Add {:?}",
                n
            ),
            Event::RemoveNode(n) => assert!(self.nodes.remove(&n.id).is_some(), "Remove {:?}", n),
            Event::AddEdge(e) => {
                assert!(self.nodes.contains_key(&e.from), "Add {:?}", e);
                assert!(self.nodes.contains_key(&e.to), "Add {:?}", e);
                assert!(self.edges.insert(e.clone()), "Add {:?}", e);
            }
            Event::RemoveEdge(e) => {
                assert!(self.nodes.contains_key(&e.from), "Remove {:?}", e);
                assert!(self.nodes.contains_key(&e.to), "Remove {:?}", e);
                assert!(self.edges.remove(&e), "Remove {:?}", e)
            }
        }
    }

    fn to_vg(&self) -> Option<VisualGraph> {
        if self.nodes.is_empty() {
            None
        } else {
            let mut out = VisualGraph::new(Orientation::TopToBottom);
            let look = StyleAttr::simple();
            let mut node_map = HashMap::new();
            println!("{:?}", self.nodes);
            for n in &self.nodes {
                let sp0 = ShapeKind::new_box(n.1);

                let size = measure_text(n.1, None, 10, 1.0);
                let sz = Point::new(size.width as _, size.height as _);

                let elm = Element::create(sp0, look.clone(), Orientation::LeftToRight, sz);
                let handle = out.add_node(elm);
                node_map.insert(n.0, handle);
            }

            for e in &self.edges {
                let arrow = Arrow::simple("");
                out.add_edge(arrow, node_map[&e.from], node_map[&e.to]);
            }

            Some(out)
        }
    }
}

struct GraphTimeline {
    current_graph: Graph,
    current_step: usize,
    events: Vec<Event>,
}

impl GraphTimeline {
    fn new(events: EventStream) -> Self {
        let s = Self {
            current_graph: Default::default(),
            current_step: 0,
            events: events.0,
        };
        s
    }

    fn next(&mut self) {
        if self.current_step >= self.events.len() {
            return;
        }

        let e = &self.events[self.current_step];
        self.current_step += 1;
        self.current_graph.apply(e.clone());
    }

    fn prev(&mut self) {
        if self.current_step == 0 {
            return;
        }

        self.current_step -= 1;
        let e = &self.events[self.current_step];
        self.current_graph.apply(e.clone().inverse());
    }

    fn go_to(&mut self, i: usize) {
        assert!(i <= self.events.len());
        match i.cmp(&self.current_step) {
            std::cmp::Ordering::Less => {
                while self.current_step != i {
                    self.prev()
                }
            }
            std::cmp::Ordering::Equal => {}
            std::cmp::Ordering::Greater => {
                while self.current_step != i {
                    self.next()
                }
            }
        }
    }
}

#[macroquad::main("BasicShapes")]
async fn main() {
    let options = CliArgs::parse();

    //let mut events = EventStream::new();
    //events.add(Event::AddNode(Node {
    //    id: 0,
    //    label: "hello".to_owned(),
    //}));
    //events.add(Event::AddNode(Node {
    //    id: 1,
    //    label: "world".to_owned(),
    //}));
    //events.add(Event::AddEdge(Edge { from: 0, to: 1 }));
    //events.add(Event::AddNode(Node {
    //    id: 2,
    //    label: "!".to_owned(),
    //}));
    //events.add(Event::AddEdge(Edge { from: 1, to: 2 }));
    //events.add(Event::AddEdge(Edge { from: 0, to: 2 }));

    //events.save(&options.input);

    let events = EventStream::load(&options.input);

    let mut timeline = GraphTimeline::new(events);

    let sh = screen_height();
    let sw = screen_width();
    let ssize = Vec2::new(sw, sh);
    let mut camera = Camera2D::from_display_rect(Rect::new(0.0, sh, sw, -sh));
    let mut actual_zoom = 1.0;

    loop {
        let md = mouse_delta_position() * ssize * 0.5 / actual_zoom;
        if is_mouse_button_down(MouseButton::Left) {
            camera.target += md;
        }
        let zoom_factor = (mouse_wheel().1 * 0.05).exp();
        camera.zoom *= zoom_factor;
        actual_zoom *= zoom_factor;

        if is_key_pressed(KeyCode::N) {
            timeline.next();
        }
        if is_key_pressed(KeyCode::P) {
            timeline.prev();
        }
        if is_key_pressed(KeyCode::Key0) {
            timeline.go_to(0)
        }
        if is_key_pressed(KeyCode::G) {
            timeline.go_to(timeline.events.len())
        }

        set_camera(&camera);

        clear_background(WHITE);

        if let Some(mut vg) = timeline.current_graph.to_vg() {
            vg.do_it(false, false, false, &mut Render);
        }

        next_frame().await
    }
}
