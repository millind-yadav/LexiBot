"use client"

import * as React from "react"
import { cn } from "@/lib/utils"

interface TabsProps {
  defaultValue?: string
  value?: string
  onValueChange?: (value: string) => void
  className?: string
  children: React.ReactNode
}

const Tabs = ({ defaultValue, value, onValueChange, className, children }: TabsProps) => {
  const [activeTab, setActiveTab] = React.useState(value || defaultValue || "")
  
  const handleValueChange = React.useCallback((newValue: string) => {
    setActiveTab(newValue)
    onValueChange?.(newValue)
  }, [onValueChange])

  React.useEffect(() => {
    if (value !== undefined) {
      setActiveTab(value)
    }
  }, [value])

  return (
    <div className={cn("w-full", className)}>
      {React.Children.map(children, (child) => {
        if (React.isValidElement(child)) {
          return React.cloneElement(child, { 
            'data-active-tab': activeTab, 
            'data-on-value-change': handleValueChange 
          } as any)
        }
        return child
      })}
    </div>
  )
}

interface TabsListProps {
  className?: string
  children: React.ReactNode
  'data-active-tab'?: string
  'data-on-value-change'?: (value: string) => void
}

const TabsList = ({ className, children, 'data-active-tab': activeTab, 'data-on-value-change': onValueChange }: TabsListProps) => (
  <div
    className={cn(
      "inline-flex h-10 items-center justify-center rounded-md bg-slate-100 p-1 text-slate-500 dark:bg-slate-800 dark:text-slate-400",
      className
    )}
  >
    {React.Children.map(children, (child) => {
      if (React.isValidElement(child)) {
        return React.cloneElement(child, { 
          'data-active-tab': activeTab, 
          'data-on-value-change': onValueChange 
        } as any)
      }
      return child
    })}
  </div>
)

interface TabsTriggerProps {
  value: string
  className?: string
  children: React.ReactNode
  'data-active-tab'?: string
  'data-on-value-change'?: (value: string) => void
}

const TabsTrigger = ({ value, className, children, 'data-active-tab': activeTab, 'data-on-value-change': onValueChange }: TabsTriggerProps) => {
  const isActive = activeTab === value
  
  return (
    <button
      className={cn(
        "inline-flex items-center justify-center whitespace-nowrap rounded-sm px-3 py-1.5 text-sm font-medium transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-slate-400 focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50",
        isActive
          ? "bg-white text-slate-950 shadow-sm dark:bg-slate-950 dark:text-slate-50"
          : "hover:bg-slate-200 hover:text-slate-900 dark:hover:bg-slate-700 dark:hover:text-slate-50",
        className
      )}
      onClick={() => onValueChange?.(value)}
    >
      {children}
    </button>
  )
}

interface TabsContentProps {
  value: string
  className?: string
  children: React.ReactNode
  'data-active-tab'?: string
}

const TabsContent = ({ value, className, children, 'data-active-tab': activeTab }: TabsContentProps) => {
  if (activeTab !== value) return null
  
  return (
    <div
      className={cn(
        "mt-2 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-slate-400 focus-visible:ring-offset-2",
        className
      )}
    >
      {children}
    </div>
  )
}

export { Tabs, TabsList, TabsTrigger, TabsContent }