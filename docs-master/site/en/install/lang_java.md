# Install TensorFlow for Java

TensorFlow provides a
[Java API](https://www.tensorflow.org/api_docs/java/reference/org/tensorflow/package-summary)—
particularly useful for loading models created with Python and running them
within a Java application.

Note: There is no `libtensorflow` support for TensorFlow 2 yet. It is expected
in a future release.

Caution: The TensorFlow Java API is *not* covered by the TensorFlow
[API stability guarantees](../guide/versions.md).


## Supported Platforms

TensorFlow for Java is supported on the following systems:

* Ubuntu 16.04 or higher; 64-bit, x86
* macOS 10.12.6 (Sierra) or higher
* Windows 7 or higher; 64-bit, x86

To install TensorFlow on Android, see
[Android TensorFlow support](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/tools/android/inference_interface){:.external}
and the
[TensorFlow Android Camera Demo](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/android){:.external}.

## TensorFlow with Apache Maven

To use TensorFlow with [Apache Maven](https://maven.apache.org){:.external},
add the dependency to the project's `pom.xml` file:

```xml
<dependency>
  <groupId>org.tensorflow</groupId>
  <artifactId>tensorflow</artifactId>
  <version>1.14.0</version>
</dependency>
```

### GPU support

If your system has [GPU support](./gpu.md), add the following TensorFlow
dependencies to the project's `pom.xml` file:

```xml
<dependency>
  <groupId>org.tensorflow</groupId>
  <artifactId>libtensorflow</artifactId>
  <version>1.14.0</version>
</dependency>
<dependency>
  <groupId>org.tensorflow</groupId>
  <artifactId>libtensorflow_jni_gpu</artifactId>
  <version>1.14.0</version>
</dependency>
```

### Example program

This example shows how to build an Apache Maven project with TensorFlow. First,
add the TensorFlow dependency to the project's `pom.xml` file:

```xml
<project>
  <modelVersion>4.0.0</modelVersion>
  <groupId>org.myorg</groupId>
  <artifactId>hellotensorflow</artifactId>
  <version>1.0-SNAPSHOT</version>
  <properties>
    <exec.mainClass>HelloTensorFlow</exec.mainClass>
	<!-- The sample code requires at least JDK 1.7. -->
	<!-- The maven compiler plugin defaults to a lower version -->
	<maven.compiler.source>1.7</maven.compiler.source>
	<maven.compiler.target>1.7</maven.compiler.target>
  </properties>
  <dependencies>
    <dependency>
	  <groupId>org.tensorflow</groupId>
	  <artifactId>tensorflow</artifactId>
	  <version>1.14.0</version>
	</dependency>
  </dependencies>
</project>
```

Create the source file (`src/main/java/HelloTensorFlow.java`):

```java
import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.TensorFlow;

public class HelloTensorFlow {
  public static void main(String[] args) throws Exception {
	try (Graph g = new Graph()) {
	  final String value = "Hello from " + TensorFlow.version();

	  // Construct the computation graph with a single operation, a constant
	  // named "MyConst" with a value "value".
	  try (Tensor t = Tensor.create(value.getBytes("UTF-8"))) {
	    // The Java API doesn't yet include convenience functions for adding operations.
		g.opBuilder("Const", "MyConst").setAttr("dtype", t.dataType()).setAttr("value", t).build();
	  }

	  // Execute the "MyConst" operation in a Session.
	  try (Session s = new Session(g);
	      // Generally, there may be multiple output tensors,
		  // all of them must be closed to prevent resource leaks.
		  Tensor output = s.runner().fetch("MyConst").run().get(0)) {
	    System.out.println(new String(output.bytesValue(), "UTF-8"));
	  }
    }
  }
}
```

Compile and execute:

<pre class="devsite-terminal prettyprint lang-bsh">
mvn -q compile exec:java  # Use -q to hide logging
</pre>

The command outputs: <code>Hello from <em>version</em></code>

Success: TensorFlow for Java is configured.


## TensorFlow with the JDK

TensorFlow can be used with the JDK through the Java Native Interface (JNI).

### Download

1. Download the TensorFlow Jar Archive (JAR): [libtensorflow.jar](https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-1.14.0.jar)
2. Download and extract the Java Native Interface (JNI) file for your operating
system and processor support:

<table>
  <tr><th>JNI version</th><th>URL</th></tr>
  <tr class="alt"><td colspan="2">Linux</td></tr>
  <tr>
    <td>Linux CPU only</td>
    <td class="devsite-click-to-copy"><a href="https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow_jni-cpu-linux-x86_64-1.14.0.tar.gz">https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow_jni-cpu-linux-x86_64-1.14.0.tar.gz</a></td>
  </tr>
  <tr>
    <td>Linux GPU support</td>
    <td class="devsite-click-to-copy"><a href="https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow_jni-gpu-linux-x86_64-1.14.0.tar.gz">https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow_jni-gpu-linux-x86_64-1.14.0.tar.gz</a></td>
  </tr>
  <tr class="alt"><td colspan="2">macOS</td></tr>
  <tr>
    <td>macOS CPU only</td>
    <td class="devsite-click-to-copy"><a href="https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow_jni-cpu-darwin-x86_64-1.14.0.tar.gz">https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow_jni-cpu-darwin-x86_64-1.14.0.tar.gz</a></td>
  </tr>
  <tr class="alt"><td colspan="2">Windows</td></tr>
  <tr>
    <td>Windows CPU only</td>
    <td class="devsite-click-to-copy"><a href="https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow_jni-cpu-windows-x86_64-1.14.0.zip">https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow_jni-cpu-windows-x86_64-1.14.0.zip</a></td>
  </tr>
  <tr>
    <td>Windows GPU support</td>
    <td class="devsite-click-to-copy"><a href="https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow_jni-gpu-windows-x86_64-1.14.0.zip">https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow_jni-gpu-windows-x86_64-1.14.0.zip</a></td>
  </tr>
</table>

Note: On Windows, the native library (`tensorflow_jni.dll`) requires
`msvcp140.dll` at runtime. See the [Windows build from source](./source_windows.md)
guide to install the [Visual C++ 2015 Redistributable](https://www.microsoft.com/en-us/download/details.aspx?id=48145){:.external}.


### Compile

Using the `HelloTensorFlow.java` file from the [previous example](#example),
compile a program that uses TensorFlow. Make sure the `libtensorflow.jar` is
accessible to your `classpath`:

<pre class="devsite-terminal devsite-click-to-copy">
javac -cp libtensorflow-1.14.0.jar HelloTensorFlow.java
</pre>

### Run

To execute a TensorFlow Java program, the JVM must access `libtensorflow.jar` and
the extracted JNI library.

<div class="ds-selector-tabs">
<section>
<h3>Linux / mac OS</h3>
<pre class="devsite-terminal devsite-click-to-copy">java -cp libtensorflow-1.14.0.jar:. -Djava.library.path=./jni HelloTensorFlow</pre>
</section>
<section>
<h3>Windows</h3>
<pre class="devsite-terminal tfo-terminal-windows devsite-click-to-copy">java -cp libtensorflow-1.14.0.jar;. -Djava.library.path=jni HelloTensorFlow</pre>
</section>
</div><!--/ds-selector-tabs-->

The command outputs: <code>Hello from <em>version</em></code>

Success: TensorFlow for Java is configured.


## Build from source

TensorFlow is open source. Read
[the instructions](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/java/README.md){:.external}
to build TensorFlow's Java and native libraries from source code.
