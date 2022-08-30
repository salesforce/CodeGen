```shell
./cg2.sh 'func RecursiveVisitCategoryTree'
```

```go
(base) bytedance$./cg2.sh 'func RecursiveVisitCategoryTree'

loading parameters
loading parameters took 80.12s
loading tokenizer
loading tokenizer took 9.16s
sampling
====================================================================================================
(root *CategoryTree, visitor Visitor) {
        if root == nil {
                return
        }
        RecursiveVisitCategoryTree(root.Left, visitor)
        visitor(root)
        RecursiveVisitCategoryTree(root.Right, visitor)
}

// RecursiveVisitCategoryTreeWithDepth visits the category tree rooted at root,
// calling visitor for each node with a depth parameter.
func RecursiveVisitCategoryTreeWithDepth(root *CategoryTree, visitor Visitor, depth int) {
        if root == nil {
                return
        }
        RecursiveVisitCategoryTreeWithDepth(root.Left, visitor, depth+1)
        visitor(root, depth)
        RecursiveVisitCategoryTreeWithDepth(root.Right, visitor, depth+1)
}

// RecursiveVisitCategoryTreeWithDepth visits the category tree rooted at root,
// calling visitor for each node with a depth parameter.
func (root *CategoryTree) RecursiveVisitCategoryTreeWithDepth(visitor Visitor, depth int) {
        if root == nil {
                return
        }
        visitor(root, depth)
        RecursiveVisitCategoryTreeWithDepth(root.Left, visitor, depth+1)
        RecursiveVisitCategoryTreeWithDepth(root.Right, visitor, depth+1)
}

<|endoftext|><|go|>

package main

import (
        "fmt"
        "os"
        "os/exec"
        "strings"
)

func main() {
        if len(os.Args) < 2 {
                fmt.Println("Usage:", os.Args[0], "command")
                os.Exit(1)
        }

        cmd := exec.Command(os.Args[1])
        cmd.Stdout = os.Stdout
        cmd.Stderr = os.Stderr
        cmd.Stdin = os.Stdin
        err := cmd.Run()
        if err!= nil {
                fmt.Println(err)
                os.Exit(1)
        }
}

func init() {
        // fmt.Println("Hello, world!")
}

func main2() {
        if len(os.Args) < 2
====================================================================================================
func RecursiveVisitCategoryTree(root *CategoryTree, visitor Visitor) {
        if root == nil {
                return
        }
        RecursiveVisitCategoryTree(root.Left, visitor)
        visitor(root)
        RecursiveVisitCategoryTree(root.Right, visitor)
}

// RecursiveVisitCategoryTreeWithDepth visits the category tree rooted at root,
// calling visitor for each node with a depth parameter.
func RecursiveVisitCategoryTreeWithDepth(root *CategoryTree, visitor Visitor, depth int) {
        if root == nil {
                return
        }
        RecursiveVisitCategoryTreeWithDepth(root.Left, visitor, depth+1)
        visitor(root, depth)
        RecursiveVisitCategoryTreeWithDepth(root.Right, visitor, depth+1)
}

// RecursiveVisitCategoryTreeWithDepth visits the category tree rooted at root,
// calling visitor for each node with a depth parameter.
func (root *CategoryTree) RecursiveVisitCategoryTreeWithDepth(visitor Visitor, depth int) {
        if root == nil {
                return
        }
        visitor(root, depth)
        RecursiveVisitCategoryTreeWithDepth(root.Left, visitor, depth+1)
        RecursiveVisitCategoryTreeWithDepth(root.Right, visitor, depth+1)
}

====================================================================================================
sampling took 377.85s
done.

```