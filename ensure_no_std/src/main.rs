// ensure_no_std/src/main.rs
#![no_std]
#![no_main]

use core::panic::PanicInfo;
use eight_point;

/// This function is called on panic.
#[panic_handler]
fn panic(_info: &PanicInfo) -> ! {
    loop {}
}

#[no_mangle]
pub extern "C" fn _start() -> ! {
    loop {}
}
