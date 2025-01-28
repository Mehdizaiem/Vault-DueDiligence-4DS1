import * as React from "react"

type CardProps = React.HTMLAttributes<HTMLDivElement>

export function Card({ className, ...props }: CardProps) {
  return (
    <div
      className={`rounded-xl border bg-white shadow ${className}`}
      {...props}
    />
  )
}