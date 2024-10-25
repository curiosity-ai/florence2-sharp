using System;

namespace Florence2;

public static class ByteFormatter
{
    public static string FormatBytes(double a, int digits = 1)
    {
        var isNegative    = a < 0;
        if (isNegative) a = -a;
        if (a == 0) return "0 B";
        const int c = 1024;
        var       e = new[] { "B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB" };
        var       f = (int)Math.Floor(Math.Log(a) / Math.Log(c));
        f = Math.Min(f, e.Length - 1);
        return (isNegative ? "-" : "") + ((a / Math.Pow(c, f)).ToString("0." + new string('#', digits))) + " " + e[f];
    }
}