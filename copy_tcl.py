import os
import shutil
import tkinter

def main():
    # 创建一个不可见的 Tk 实例来获取真实的 tcl 和 tk 路径
    root = tkinter.Tk()
    root.withdraw()
    
    tcl_lib = root.tk.exprstring('$tcl_library')
    tk_lib = root.tk.exprstring('$tk_library')
    
    root.destroy()
    
    print(f"真实的 TCL_LIBRARY: {tcl_lib}")
    print(f"真实的 TK_LIBRARY: {tk_lib}")
    
    # 目标目录
    dest_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tcl_tk_libs")
    if os.path.exists(dest_dir):
        shutil.rmtree(dest_dir)
    os.makedirs(dest_dir)
    
    dest_tcl = os.path.join(dest_dir, "tcl")
    dest_tk = os.path.join(dest_dir, "tk")
    
    print("正在复制 Tcl 库...")
    shutil.copytree(tcl_lib, dest_tcl)
    
    print("正在复制 Tk 库...")
    shutil.copytree(tk_lib, dest_tk)
    
    print("复制完成！")

if __name__ == '__main__':
    main()
