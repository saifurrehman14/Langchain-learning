from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=700 ,
    chunk_overlap=0
)
text = """
A breadboard is a widely used tool in electronics engineering that allows engineers, students, and hobbyists to design, build, and test electronic circuits without the need for soldering. It is especially useful during the prototyping phase of circuit development, where frequent changes and adjustments are required. The breadboard is typically made of a rectangular plastic board with numerous small holes arranged in a grid pattern. Underneath these holes are conductive metal strips that create electrical connections between specific rows or columns, enabling components to be easily inserted and connected.

The structure of a breadboard is divided into two main sections: the terminal strips and the power rails. The terminal strips are located in the central area and are used for placing electronic components such as resistors, capacitors, diodes, and integrated circuits. Each column of holes in this section is internally connected, usually in groups of five, allowing multiple component leads to share the same electrical node. A gap runs down the middle of the breadboard, separating the two halves of the terminal strip. This gap is designed to accommodate dual in-line package (DIP) integrated circuits, ensuring that pins on opposite sides are not shorted together.

On either side of the breadboard are the power rails, which are used to supply voltage and ground to the circuit. These rails run vertically and are typically marked with red and blue lines to indicate positive and negative connections. Engineers connect a power supply to these rails and then distribute power to different parts of the circuit using jumper wires. This organized layout helps in maintaining clarity and reducing wiring complexity.

One of the main advantages of a breadboard is its reusability. Components can be inserted and removed multiple times without causing damage, making it a cost-effective tool for experimentation and learning. It also allows for rapid prototyping, as circuits can be assembled quickly and modified easily. This flexibility is particularly important in the design process, where testing and debugging are essential steps.

However, breadboards also have some limitations. They are not suitable for high-frequency or high-power circuits because the loose connections and internal capacitance can introduce noise and signal distortion. Additionally, improper wiring or loose component placement can lead to unreliable connections, which may cause circuit malfunction. Therefore, careful organization and proper handling are necessary when using a breadboard.

In educational environments, breadboards are commonly used to teach fundamental concepts such as circuit analysis, Ohm’s law, and digital logic design. They provide a hands-on learning experience that helps students better understand theoretical concepts. In professional settings, engineers use breadboards to verify circuit functionality before transferring the design to a printed circuit board (PCB) for permanent implementation.

In conclusion, the breadboard is an essential tool in electronics engineering that offers a simple, flexible, and efficient way to prototype and test circuits. Despite the availability of advanced simulation tools, it remains a valuable resource for practical experimentation and innovation.

"""
result = splitter.split_text(text)
print(result[0])
print("-----------------------")
print(result[1])
print("-----------------------")

print(result[2])
print("-----------------------")

print(result[3])