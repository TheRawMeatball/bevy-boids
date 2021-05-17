#![feature(array_map)]
use bevy::{prelude::*, render::camera::OrthographicProjection};

// Boids is an artificial life program which simulates the flocking behaviour of birds.
// See https://en.wikipedia.org/wiki/Boids
fn main() {
    let mut app = App::build();
    app.insert_resource(Msaa { samples: 4 })
        .add_plugins(DefaultPlugins);
    app.add_startup_system(setup.system())
        .add_system(boids_flocking_system.system().label("flocking"))
        .add_system(
            boid_heading_system
                .system()
                .label("heading")
                .after("flocking"),
        )
        .add_system(ls_adjustment.system().before("flocking"))
        .add_system(heading_system.system().after("heading"))
        .run();
}

// These constants influence how the boids are spawned into the scene
const NUM_BOIDS: usize = 300; // The number of boids we will spawn

// These constants influence the boid movement
const MAX_VELOCITY: f32 = 16.;
const MIN_VELOCITY: f32 = 4.;
const MAX_ACCELERATION: f32 = 6.;
const NEIGHBOR_RADIUS: f32 = 20.;
const FIELD_OF_VISION: f32 = (2. / 3.) * std::f32::consts::TAU;
const AVOID_RADIUS: f32 = 5.;
const INTERFLOCK_SEPARATION_FORCE: f32 = 15.;
const SEPARATION_FORCE: f32 = 10.;
const ALIGN_FORCE: f32 = 1.;
const COHESION_FORCE: f32 = 1.;
const TARGET_FORCE: f32 = 0.001;
const COLLISION_AVOIDANCE_FORCE: f32 = 20.;
const COLLISION_RADIUS: f32 = 10.;

const TURN_FIND_STEP: f32 = (std::f32::consts::PI / 180.) * 45.;

// Heading component (velocity vector)
#[derive(Default)]
pub struct Velocity(pub Vec2);

impl std::ops::Deref for Velocity {
    type Target = Vec2;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

fn heading_system(time: Res<Time>, mut query: Query<(&Velocity, &mut Transform)>) {
    query.for_each_mut(|(heading, mut transform)| {
        transform.translation += heading.0.extend(0.) * time.delta_seconds();
    });
}

// Boid component
#[derive(Default)]
pub struct Boid {
    pub force: Vec2, // Sum of the forces
}

#[derive(Default, PartialEq, Eq)]
pub struct Flock(usize);

fn steer(current_vel: Vec2, dir: Vec2) -> Vec2 {
    (dir * MAX_VELOCITY - current_vel).clamp_length_max(MAX_ACCELERATION)
}

fn percieve(pos: Vec2, dir: Vec2, other: Vec2) -> bool {
    pos.distance_squared(other) < NEIGHBOR_RADIUS * NEIGHBOR_RADIUS
        && (pos + dir).angle_between(other) < FIELD_OF_VISION / 2.
}

// System to calculate each boids forces
fn boids_flocking_system(
    mut query: Query<(&mut Boid, &Transform, &Velocity, Entity, &Flock)>,
    inner_query: Query<(&Transform, &Velocity, Entity, &Flock), With<Boid>>, // This query requires the Boid component but doesn't borrow it
    raycastable: Query<&Raycastable>,
) {
    query.for_each_mut(|(mut boid, transform_a, velocity_a, entity_a, flock_a)| {
        let pos_a = transform_a.translation.truncate();
        let velocity_a = velocity_a.0;
        let dir_a = velocity_a.normalize();
        let mut acceleration = Vec2::ZERO;
        let mut flockmate_count = 0;
        let mut foreign_count = 0;
        let mut pos_total = Vec2::ZERO;
        let mut heading_total = Vec2::ZERO;
        let mut avoidance_total = Vec2::ZERO;
        let mut interflock_avoidance_total = Vec2::ZERO;

        inner_query.for_each(|(transform_b, velocity_b, entity_b, flock_b)| {
            if entity_a == entity_b {
                return;
            }
            let pos_b = transform_b.translation.truncate();
            let direction_b = velocity_b.normalize();
            if percieve(pos_a, dir_a, pos_b) {
                let offset = pos_b - pos_a;
                let sqr_dist = offset.length_squared();
                if flock_a == flock_b {
                    flockmate_count += 1;
                    pos_total += pos_b;
                    heading_total += direction_b;
                    if sqr_dist < AVOID_RADIUS * AVOID_RADIUS {
                        avoidance_total -= offset / sqr_dist;
                    }
                } else {
                    foreign_count += 1;
                    interflock_avoidance_total -= offset / sqr_dist;
                }
            }
        });

        if foreign_count > 0 {
            let foreign_count = foreign_count as f32;
            let avg_interflock_avoidance = interflock_avoidance_total / foreign_count;
            acceleration +=
                steer(velocity_a, avg_interflock_avoidance) * INTERFLOCK_SEPARATION_FORCE;
        }

        if flockmate_count > 0 {
            let flockmate_count = flockmate_count as f32;
            let avg_pos = pos_total / flockmate_count;
            let avg_avoidance = avoidance_total / flockmate_count;
            let avg_heading = heading_total / flockmate_count;
            let offset_to_avg_pos = avg_pos - pos_a;

            acceleration += steer(velocity_a, avg_avoidance) * SEPARATION_FORCE;
            acceleration += steer(velocity_a, offset_to_avg_pos) * COHESION_FORCE;
            acceleration += steer(velocity_a, avg_heading) * ALIGN_FORCE;
        }
        let target_pos = Vec2::ZERO;
        acceleration += steer(velocity_a, target_pos - pos_a) * TARGET_FORCE;

        if raycast(raycastable.iter(), pos_a, dir_a * COLLISION_RADIUS) {
            let dir_angle = f32::atan2(dir_a.y, dir_a.x);
            let direction = (0..(std::f32::consts::TAU / TURN_FIND_STEP) as usize)
                .map(|i| {
                    if i % 2 == 0 {
                        TURN_FIND_STEP * (i / 2) as f32
                    } else {
                        -TURN_FIND_STEP * ((i + 1) / 2) as f32
                    }
                })
                .map(|v| v + dir_angle)
                .map(vec_from_angle)
                .find(|&dir| !raycast(raycastable.iter(), pos_a, dir * COLLISION_RADIUS))
                .unwrap_or(dir_a);

            acceleration += steer(velocity_a, direction) * COLLISION_AVOIDANCE_FORCE;
        }

        boid.force = acceleration;
    });
}
fn vec_from_angle(x: f32) -> Vec2 {
    Vec2::new(f32::cos(x), f32::sin(x))
}

fn raycast<'a>(
    raycastable: impl IntoIterator<Item = &'a Raycastable>,
    start: Vec2,
    offset: Vec2,
) -> bool {
    raycastable.into_iter().any(|r| match r {
        Raycastable::LS(ls) => segment_segment_intersection(*ls, LineSegment { start, offset }),
    })
}

struct MainCamera;

// System to apply the boid force to the heading/rotation component
fn boid_heading_system(
    time: Res<Time>,
    mut query: Query<(&Boid, &mut Velocity, &mut Transform)>,
    cam: Query<&OrthographicProjection, With<MainCamera>>,
) {
    query.for_each_mut(|(boid, mut velocity, mut transform)| {
        let rotation = &mut transform.rotation;

        // Update the heading (velocity)
        velocity.0 += boid.force * time.delta_seconds();
        velocity.0 = velocity.0.clamp_length(MIN_VELOCITY, MAX_VELOCITY);

        // Compute the rotation according to the heading
        let angle = f32::atan2(velocity.y, velocity.x);
        *rotation = Quat::from_rotation_ypr(0., 0., angle);

        let pos = &mut transform.translation;
        let proj = cam.single().unwrap();
        if pos.x < proj.left * proj.scale {
            pos.x = proj.right * proj.scale
        }
        if pos.x > proj.right * proj.scale {
            pos.x = proj.left * proj.scale
        }
        if pos.y < proj.bottom * proj.scale {
            pos.y = proj.top * proj.scale
        }
        if pos.y > proj.top * proj.scale {
            pos.y = proj.bottom * proj.scale
        }
    });
}

// Set up a scene
fn setup(
    mut commands: Commands,
    mut materials: ResMut<Assets<ColorMaterial>>,
    server: Res<AssetServer>,
) {
    commands
        // Camera, follows the last boid
        .spawn_bundle(OrthographicCameraBundle {
            orthographic_projection: OrthographicProjection {
                scale: 40.,
                scaling_mode: bevy::render::camera::ScalingMode::FixedVertical,
                ..Default::default()
            },
            ..OrthographicCameraBundle::new_2d()
        })
        .insert(MainCamera);

    let handle = server.load("icon.png");
    let materials = [
        Color::RED,
        Color::GREEN,
        Color::BLUE,
        Color::GOLD,
        Color::PINK,
        Color::WHITE,
        Color::PURPLE,
        Color::TEAL,
        Color::BEIGE,
    ]
    .map(|v| ColorMaterial::modulated_texture(handle.clone(), v))
    .map(|v| materials.add(v));
    #[derive(Default, Bundle)]
    struct GroupBundle {
        #[bundle]
        sprite: SpriteBundle,
        velocity: Velocity,
        boid: Boid,
        flock: Flock,
    }
    let random_pm = |v: f32| rand::random::<f32>() * 2. * v - v;
    commands.spawn_batch((0..NUM_BOIDS).map(move |i| {
        let handle = materials[i % materials.len()].clone();
        GroupBundle {
            sprite: SpriteBundle {
                transform: Transform::from_xyz(random_pm(10.), random_pm(10.), 0.),
                sprite: Sprite {
                    resize_mode: SpriteResizeMode::Manual,
                    size: Vec2::ONE * 1.,
                    ..Default::default()
                },
                material: handle,
                ..Default::default()
            },
            velocity: Velocity({
                let angle = rand::random::<f32>() * std::f32::consts::TAU;
                Vec2::new(angle.cos(), angle.sin()) * MIN_VELOCITY
            }),
            flock: Flock(i % materials.len()),
            ..Default::default()
        }
    }));

    commands
        .spawn()
        .insert(Raycastable::LS(LineSegment::default()))
        .insert(Left);
    commands
        .spawn()
        .insert(Raycastable::LS(LineSegment::default()))
        .insert(Right);
    commands
        .spawn()
        .insert(Raycastable::LS(LineSegment::default()))
        .insert(Top);
    commands
        .spawn()
        .insert(Raycastable::LS(LineSegment::default()))
        .insert(Bottom);
}

fn ls_adjustment(
    mut left: Query<&mut Raycastable, (With<Left>, Without<Right>, Without<Top>, Without<Bottom>)>,
    mut right: Query<&mut Raycastable, (Without<Left>, With<Right>, Without<Top>, Without<Bottom>)>,
    mut top: Query<&mut Raycastable, (Without<Left>, Without<Right>, With<Top>, Without<Bottom>)>,
    mut bottom: Query<
        &mut Raycastable,
        (Without<Left>, Without<Right>, Without<Top>, With<Bottom>),
    >,
    cam: Query<&OrthographicProjection, With<MainCamera>>,
) {
    let Raycastable::LS(left) = &mut *left.single_mut().unwrap();
    let Raycastable::LS(right) = &mut *right.single_mut().unwrap();
    let Raycastable::LS(top) = &mut *top.single_mut().unwrap();
    let Raycastable::LS(bottom) = &mut *bottom.single_mut().unwrap();

    let cam = cam.single().unwrap();
    *left = LineSegment {
        start: Vec2::new(cam.left, cam.top) * cam.scale,
        offset: Vec2::Y * -2. * cam.scale,
    };
    *right = LineSegment {
        start: Vec2::new(cam.right, cam.top) * cam.scale,
        offset: Vec2::Y * -2. * cam.scale,
    };
    *top = LineSegment {
        start: Vec2::new(cam.left, cam.top) * cam.scale,
        offset: Vec2::X * (cam.right - cam.left) * cam.scale,
    };
    *bottom = LineSegment {
        start: Vec2::new(cam.left, cam.bottom) * cam.scale,
        offset: Vec2::X * (cam.right - cam.left) * cam.scale,
    };
}

struct Left;
struct Right;
struct Top;
struct Bottom;

enum Raycastable {
    LS(LineSegment),
}

#[derive(Clone, Copy, Debug, Default)]
struct LineSegment {
    start: Vec2,
    offset: Vec2,
}

// https://stackoverflow.com/a/565282
fn segment_segment_intersection(p: LineSegment, q: LineSegment) -> bool {
    let LineSegment {
        start: p,
        offset: r,
    } = p;
    let LineSegment {
        start: q,
        offset: s,
    } = q;
    let r_x_s = r.perp_dot(s);
    let q_minus_p = q - p;
    if r_x_s == 0. {
        let qmp_x_r = q_minus_p.perp_dot(r);
        if qmp_x_r == 0. {
            // collinear
            let r_dot_r = r.dot(r);
            let r_over_rdr = r / r_dot_r;
            let t0 = q_minus_p.dot(r_over_rdr);
            let t1 = t0 + s.dot(r_over_rdr);
            if ((0.0..1.0).contains(&t0) || (0.0..1.0).contains(&t1))
                || (t0 < 0. && t1 > 1.)
                || (t0 > 1. && t0 < 0.)
            {
                // overlapping
                true
            } else {
                // disjoint
                false
            }
        } else {
            // parallel & non-intersecting
            false
        }
    } else {
        let t = q_minus_p.perp_dot(s / r_x_s);
        let u = q_minus_p.perp_dot(r / r_x_s);
        if 0. <= t && t <= 1. && 0. <= u && u <= 1. {
            // intersection @ (p + tr, q + us)
            true
        } else {
            // no intersection within segments
            false
        }
    }
}
