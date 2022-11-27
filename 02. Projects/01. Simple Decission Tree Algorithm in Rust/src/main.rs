use linfa::prelude::*;
use linfa_trees::DecisionTree;
use linfa_trees::SplitQuality;
use ndarray::prelude::*;
use ndarray::Array2;
use std::fs::File;
use std::io::Write;

fn main() {
    let original_data: Array2<f32> = array![
        [1., 1., 0., 1., 10.],
        [1., 0., 1., 1., 6.],
        [1., 1., 1., 0., 8.],
        [1., 0., 0., 1., 4.],
        [1., 0., 0., 100., 7.],
        [1., 1., 0., 0., 9.],
        [1., 0., 0., 600., 5.],
        [1., 1., 0., 0., 8.],
        [1., 0., 0., 500., 6.],
        [1., 1., 0., 0., 9.],
        [1., 0., 0., 900., 5.],
        [1., 1., 0., 0., 8.],
        [1., 0., 0., 400., 6.],
        [1., 1., 0., 0., 9.],
        [1., 0., 0., 0., 5.],
        [1., 1., 0., 600., 8.],
        [1., 0., 0., 0., 6.],
        [1., 1., 0., 800., 9.],
        [1., 0., 0., 0., 5.],
        [1., 1., 0., 300., 8.],
        [1., 0., 0., 0., 6.],
        [1., 1., 0., 0., 9.],
        [1., 0., 0., 0., 5.],
        [1., 1., 0., 600., 8.],
        [1., 0., 0., 0., 6.],
        [1., 1., 0., 500., 9.],
        [1., 0., 0., 0., 5.],
        [1., 1., 0., 1000., 8.],
        [1., 0., 0., 0., 6.],
        [1., 1., 0., 0., 9.],
        [1., 0., 0., 300., 5.],
        [1., 1., 0., 800., 8.],
        [1., 0., 0., 50., 6.],
        [1., 1., 0., 600., 9.],
    ];

    let feature_name = vec!["Watched BEN10", "Swimming", "Ate Pizza", "Rust LOC"];

    let num_features = original_data.len_of(Axis(1)) - 1;

    let features = original_data.slice(s![.., 0..num_features]).to_owned();

    let labels = original_data.column(num_features).to_owned();

    let linfa_dataset = Dataset::new(features, labels)
        .map_targets(|x| match x.to_owned() as i32 {
            i32::MIN..=4 => "Sad",
            5..=7 => "Ok",
            8..=i32::MAX => "Happy",
        })
        .with_feature_names(feature_name);

    let model = DecisionTree::params()
        .split_quality(SplitQuality::Gini)
        .fit(&linfa_dataset)
        .unwrap();

    File::create("dt.tex")
        .unwrap()
        .write_all(model.export_to_tikz().with_legend().to_string().as_bytes())
        .unwrap();
}
