import pygetwindow as gw


def minimize_window_by_name(partial_name):
    windows = gw.getWindowsWithTitle(partial_name)
    if windows:
        for win in windows:
            if partial_name.lower() in win.title.lower():
                win.minimize()
                print(f"Minimized: {win.title}")
                return
    print(f"No window found containing '{partial_name}'.")

# Example usage
minimize_window_by_name("BlueMuse.exe")