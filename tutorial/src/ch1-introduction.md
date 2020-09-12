## An introduction to Rust-CV

Rust-CV is a project that aims to bring Computer Vision (CV) algorithms to the Rust programming language. To follow this tutorial a basic understanding of Rust, and its ecosystem, is required. Also, computer vision knowledge would help.

### About this book

This tutorial aims to help you understand how to use Rust-CV and what it has to offer. In is current state, this tutorial is rather incomplete and a lot of examples are missing. If you spot an error or wish to add a page feel free to do a PR. We are very open so don't hesitate to contribute.

### Project structure

Before using Rust-CV it is important to understand how the ecosystem is set-up and how to use it.

If you look at the [repository](https://github.com/rust-cv/cv) you can see multiples directories and a `Cargo.toml` file that contains a `[workspace]` section. As the name implies, we are using the workspace feature of Cargo. To put it simply, a cargo workspace allows us to creates multiples package in the same repository. 
You can check the [official documentation](https://doc.rust-lang.org/book/ch14-03-cargo-workspaces.html) to get more details.

This is what happens here. Rust-CV is a mono-repository project which is made of multiples packages. For instance, we have the `cv-core` directory that contains the very basic things that for computer vision. We also have the `imgshow` directory which allow us to show images. There is many more but we won't go deeper for now. The `cv` crate needs to be explained though. The `cv` crate is rather empty (code wise) and just reexport most of the other package so by just depending of it we have most of what the project has to offer.

There is three things to remember here :
* [Rust-CV repository](https://github.com/rust-cv/cv) is a mono repository that is split-up into multiples packages.
* The way the project is structured allow you to use tiny crates so you don't have to pay the price for all the CV code if you just use a subpart. 
* The project define a crate named `cv` that depends on many others just to re-export them. This is useful to get started faster as just by pulling this crate you have already a lot of computer vision algorithm and data structure ready to use.