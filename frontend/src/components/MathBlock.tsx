import katex from "katex";

type MathBlockProps = {
  children: string;
  block?: boolean;
};

export function MathBlock({ children, block = false }: MathBlockProps) {
  const html = katex.renderToString(children, {
    displayMode: block,
    throwOnError: false,
    strict: "ignore"
  });

  return (
    <span
      className={block ? "math-block" : "math-inline"}
      dangerouslySetInnerHTML={{ __html: html }}
    />
  );
}
