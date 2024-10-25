using System.IO;
using System.Reflection;

namespace Florence2;

public static class ResourceLoader
{
    public static Stream OpenResource(Assembly assembly, string resourceFile)
    {
        return assembly.GetManifestResourceStream(assembly.GetName().Name + ".Resources." + resourceFile);
    }

    public static byte[] GetResource(Assembly assembly, string resourceFile)
    {
        var s  = assembly.GetManifestResourceStream(assembly.GetName().Name + ".Resources." + resourceFile);
        var b  = new byte[s.Length];
        var ms = new MemoryStream(b);
        s.CopyTo(ms);
        return b;
    }
}